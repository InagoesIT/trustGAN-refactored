# Authors:
#   Helion du Mas des Bourboux <helion.dumasdesbourboux'at'thalesgroup.com>
#
# MIT License
#
# Copyright (c) 2022 THALES
#   All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# 2022 october 21

import os
import glob
import numpy as np
import torch
import time

from py.dataset.modifier import Modifier
from py.performances.performances_logger import PerformancesLogger
from py.training.training_data.networks_data import NetworksData
from py.training.training_data.state import State
from py.training.training_data.data_loaders import DataLoaders
from py.training.training_data.parameters import Parameters
from py.training.network_nan_recovery import NetworkNaNRecovery
from py.training.training_data.paths import Paths
from py.utils.plotter import Plotter
from py.utils.saver import Saver


class TrainingPipeline:
    def __init__(
            self,
            parameters: Parameters,
            paths: Paths,
            state: State
    ):
        self.paths = paths
        self.parameters = parameters
        self.state = state
        self.execution_data_file_name = "{}/execution_data.npy"

        self.saver = None
        self.plotter = None
        self.models_data = None

        TrainingPipeline.set_seeds(state.seed)
        self.data_loaders = DataLoaders(
            path_to_dataset=paths.dataset, training_parameters=parameters)
        self.performances_logger = None

        self.modifier = Modifier(nr_channels=self.parameters.nr_channels)
        self.nr_dimensions = self.modifier(next(iter(self.data_loaders.validation[0])))[0].ndim - 2
        self.parameters.nr_channels = self.modifier(next(iter(self.data_loaders.validation[0])))[0].shape[1]
        print(f"INFO: Found {self.nr_dimensions} dimensions")
        print(f"INFO: Found {self.parameters.nr_channels} channels")

    @staticmethod
    def set_seeds(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_models_if_present(self):
        for model, path_to_load in [
            (self.models_data.target_model, self.paths.load_target_model),
            (self.models_data.gan, self.paths.load_target_model),
        ]:
            model = model.to(self.state.device)

            if path_to_load is not None:
                ld = torch.load(path_to_load, map_location=self.state.device)
                model.load_state_dict(ld)

    def target_model_train(self, inputs, labels):
        self.models_data.target_model.run()
        self.models_data.target_model_optimizer.zero_grad()

        target_model_outputs = self.models_data.target_model(inputs)

        loss_target_model = self.models_data.target_model_loss_type(target_model_outputs, labels)
        loss_target_model.backward()

        torch.nn.utils.clip_grad_norm_(self.models_data.target_model.parameters(),
                                       self.models_data.grad_clipping_coeff)
        self.models_data.target_model_optimizer.step()

        _, truth = torch.max(labels, 1)
        _, predicted = torch.max(target_model_outputs, 1)
        accuracies_target_model = (predicted == truth).float().mean()

        return loss_target_model, accuracies_target_model

    def target_model_on_gan_train(self, inputs_shape, labels_shape):
        self.models_data.target_model.run()
        self.models_data.target_model_optimizer.zero_grad()

        rand_inputs = torch.rand(inputs_shape, device=self.state.device)
        rand_labels = (
                1.0 / self.parameters.nr_classes * torch.ones(labels_shape, device=self.state.device)
        )

        explore_probability = torch.rand(1)
        networks = glob.glob("{}/nets/gan-*-step-*.pth".format(self.paths.root_folder))

        # load a previous gan model or use the current one
        gan_to_use = self.models_data.gan
        if (explore_probability < 0.1) and (len(networks) > 0):
            nr_network = torch.randint(low=0, high=len(networks), size=[1])
            self.models_data.gan_copy.load_state_dict(torch.load(networks[nr_network]))
            gan_to_use = self.models_data.gan_copy

        # get the generated output of the gan (don't run it!)
        gan_to_use.eval()
        gan_outputs = gan_to_use(rand_inputs)
        gan_to_use.run()

        # backprop the loss
        target_model_outputs = self.models_data.target_model(gan_outputs)
        loss_target_model_on_gan = self.models_data.target_model_loss_type(target_model_outputs, rand_labels)
        loss_target_model_on_gan.backward()
        torch.nn.utils.clip_grad_norm_(self.models_data.target_model.parameters(),
                                       self.models_data.grad_clipping_coeff)
        self.models_data.target_model_optimizer.step()

        return loss_target_model_on_gan

    def gan_train(self, inputs_shape):
        self.models_data.gan.train()
        self.models_data.gan_optimizer.zero_grad()
        rand_inputs = torch.rand(inputs_shape, device=self.state.device)

        gan_outputs = self.models_data.gan(rand_inputs)
        self.models_data.target_model.eval()

        target_model_outputs = self.models_data.target_model(gan_outputs)
        self.models_data.target_model.train()

        loss_gan = self.models_data.gan_loss_type(rand_inputs.float(), gan_outputs, target_model_outputs)
        loss_gan.backward()

        self.state.loss_gan = loss_gan.item()
        if self.state.loss_gan < self.state.best_loss:
            self.saver.save_epoch(
                best="is_best",
                gan_outputs=gan_outputs,
                target_model_outputs=target_model_outputs,
                save_plot=True,
            )

            self.state.best_loss = self.state.loss_gan

        torch.nn.utils.clip_grad_norm_(self.models_data.gan.parameters(), 1.0)
        self.models_data.gan_optimizer.step()

        return loss_gan

    def recover_from_nan(self, model_index):
        nan_recovery = NetworkNaNRecovery(self.models_data, self.paths.root_folder, self.state.device,
                                          self.data_loaders.train[model_index], self.modifier)
        nan_recovery.recover_from_nan_target_model()
        nan_recovery.recover_from_nan_gan()

    def is_gan_training_epoch(self):
        proportion_target_model_alone = torch.rand(1)
        return self.state.epoch >= self.parameters.nr_steps_target_model_alone and \
               proportion_target_model_alone > self.parameters.proportion_target_model_alone

    def train_models(self, model_index):
        for i, data in enumerate(self.data_loaders.train[model_index]):
            inputs, labels = data[0].to(self.state.device), data[1].to(self.state.device)
            inputs, labels = self.modifier((inputs, labels))

            if self.is_gan_training_epoch():
                for _ in range(self.parameters.nr_steps_gan):
                    self.state.loss_gan = self.gan_train(list(inputs.shape))

                for _ in range(self.parameters.nr_steps_target_model_gan):
                    loss_target_model_on_gan = self.target_model_on_gan_train(
                        list(inputs.shape), list(labels.shape))
            else:
                loss_target_model_on_gan = -1.0
                self.state.loss_gan = -1.0

            loss_target_model, accuracies_target_model = self.target_model_train(inputs, labels)

            if i % 10 == 0:
                print(
                    f"network index: {model_index}; "
                    f"data index: {i:03d}/{len(self.data_loaders.train[model_index]):03d}, "
                    f"Loss: net = {loss_target_model:6.3f}, net_on_gan = {loss_target_model_on_gan:6.3f}, "
                    f"gan = {self.loss_gan:6.3f}, "
                    f"Accs: net = {accuracies_target_model:6.3f}"
                )

    def log_execution_time(self, start_time, model_index):
        if model_index == 0:
            self.state.execution_data["time"] += [(time.time() - start_time) / 60]
        else:
            self.state.execution_data["time"][self.state.epoch] += (time.time() - start_time) / 60
            self.state.execution_data["time"][self.state.epoch] /= 2

    def set_training_mode(self):
        self.models_data.target_model.train()
        self.models_data.gan.train()

    def train_model_with_index(self, model_index):
        for self.epoch in range(self.parameters.total_epochs):
            start_time = time.time()
            self.recover_from_nan(model_index)

            self.performances_logger.run(model_index)
            self.saver.save_best_validation_loss()
            self.state.best_loss = float("inf")

            self.set_training_mode()
            self.train_models(model_index)

            self.saver.save_model_data()
            self.log_execution_time(start_time, model_index)

        if self.state.loss_gan != -1.0:
            self.models_data.gan_scheduler.step()
        self.performances_logger.run(model_index)

    def initialize_data_for_new_model(self):
        self.state.epoch = 0
        self.models_data = NetworksData(self.state.given_target_model, self.nr_dimensions, self.parameters)
        self.plotter = Plotter(device=self.state.device, modifier=self.modifier,
                               networks_data=self.models_data, root_folder=self.paths.root_folder,
                               total_epochs=self.parameters.total_epochs)
        self.saver = Saver(device=self.state.device, modifier=self.modifier, networks_data=self.models_data,
                           plotter=self.plotter, root_folder=self.paths.root_folder)
        self.performances_logger = PerformancesLogger(self)

        self.models_data.gan = self.models_data.gan.to(self.state.device)
        self.load_models_if_present()

    def load_logs(self):
        if os.path.exists("{}/performances.npy".format(self.paths.root_folder)):
            self.state.perfs = np.load("{}/performances.npy".format(self.paths.root_folder), allow_pickle=True)
            self.state.perfs = self.state.perfs.item()

        if os.path.exists(self.execution_data_file_name.format(self.paths.root_folder)):
            self.state.execution_data = np.load(self.execution_data_file_name.format(self.paths.root_folder),
                                                allow_pickle=True)
            self.state.execution_data = self.state.execution_data.item()

    def initialize_state(self):
        self.state.perfs = {"run": {}, "valid": {}}
        self.state.perfs["run"]["is_best-gan-loss"] = []
        self.state.perfs["valid"]["is_best-gan-loss"] = []
        self.state.best_loss = float("inf")

    def log_execution_data(self):
        self.state.execution_data["memory"] = [torch.cuda.max_memory_allocated(0)]
        np.save(self.execution_data_file_name.format(self.paths.root_folder), self.state.execution_data)

    def run(self):
        self.initialize_state()
        self.load_logs()

        for model_index in range(self.parameters.k_fold):
            self.initialize_data_for_new_model()
            self.train_model_with_index(model_index)

        self.log_execution_data()
