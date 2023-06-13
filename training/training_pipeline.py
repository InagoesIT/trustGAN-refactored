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

import glob
import numpy as np
import torch
import time

from dataset.modifier import Modifier
from performances.performances_logger import PerformancesLogger
from training.components.networks_data import NetworksData
from training.components.state import State
from training.components.data_loaders import DataLoaders
from training.components.hyperparameters import Hyperparameters
from training.network_nan_recovery import NetworkNaNRecovery
from training.components.paths import Paths
from utils.saver import Saver


class TrainingPipeline:
    def __init__(
            self,
            hyperparameters: Hyperparameters,
            paths: Paths,
            state: State
    ):
        self.paths = paths
        self.hyperparameters = hyperparameters
        self.state = state
        self.execution_data_file_name = "{}/execution_data.npy"
        self.saver = None
        self.networks_data = None

        TrainingPipeline.set_seeds(state.seed)
        self.data_loaders = DataLoaders(path_to_dataset=paths.dataset,
                                        nr_classes=self.state.nr_classes,
                                        training_hyperparameters=hyperparameters)
        self.performances_logger = None

        self.modifier = Modifier(nr_channels=self.hyperparameters.nr_channels)
        self.state.nr_dimensions = self.modifier(next(iter(self.data_loaders.validation[0])))[0].ndim - 2
        self.hyperparameters.nr_channels = self.modifier(next(iter(self.data_loaders.validation[0])))[0].shape[1]
        print(f"INFO: Found {self.state.nr_dimensions} dimensions")
        print(f"INFO: Found {self.hyperparameters.nr_channels} channels")

    @staticmethod
    def set_seeds(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_networks_outputs(self, loader):
        self.networks_data.target_model.eval()
        self.networks_data.gan.eval()
        dims = list(self.modifier(next(iter(loader)))[0].shape)
        rand_inputs = torch.rand(dims, device=self.state.device)

        gan_outputs = self.networks_data.gan(rand_inputs)
        target_model_outputs = self.networks_data.target_model(gan_outputs)
        self.networks_data.target_model.train()
        self.networks_data.gan.train()

        return gan_outputs, target_model_outputs

    def target_model_train(self, inputs, labels):
        self.networks_data.target_model.train()
        self.networks_data.target_model_optimizer.zero_grad()

        target_model_outputs = self.networks_data.target_model(inputs)
        loss_target_model = self.networks_data.target_model_loss_function(target_model_outputs, labels)
        loss_target_model.backward()

        torch.nn.utils.clip_grad_norm_(self.networks_data.target_model.parameters(),
                                       self.networks_data.gradient_clipping_coefficient)
        self.networks_data.target_model_optimizer.step()

        _, truth = torch.max(labels, 1)
        _, predicted = torch.max(target_model_outputs, 1)
        accuracies_target_model = (predicted == truth).float().mean()

        return loss_target_model, accuracies_target_model

    def target_model_on_gan_train(self, inputs_shape, labels_shape):
        self.networks_data.target_model.train()
        self.networks_data.target_model_optimizer.zero_grad()

        rand_inputs = torch.rand(inputs_shape, device=self.state.device)
        rand_labels = (
                1.0 / self.state.nr_classes * torch.ones(labels_shape, device=self.state.device)
        )

        explore_probability = torch.rand(1)
        networks = glob.glob("{}/networks/gan-*-step-*.pth".format(self.paths.root_folder))

        # load a previous gan model or use the current one
        gan_to_use = self.networks_data.gan
        if (explore_probability < 0.1) and len(networks) > 0:
            nr_network = torch.randint(low=0, high=len(networks), size=[1])
            self.networks_data.gan_copy.load_state_dict(torch.load(networks[nr_network]))
            gan_to_use = self.networks_data.gan_copy

        # get the generated output of the gan (don't train it!)
        gan_to_use.eval()
        gan_outputs = gan_to_use(rand_inputs)
        gan_to_use.train()

        # backprop the loss
        target_model_outputs = self.networks_data.target_model(gan_outputs)
        loss_target_model_on_gan = self.networks_data.target_model_on_gan_loss_function(target_model_outputs,
                                                                                        rand_labels)
        loss_target_model_on_gan.backward()
        torch.nn.utils.clip_grad_norm_(self.networks_data.target_model.parameters(),
                                       self.networks_data.gradient_clipping_coefficient)
        self.networks_data.target_model_optimizer.step()

        return loss_target_model_on_gan

    def gan_train(self, inputs_shape):
        self.networks_data.gan.train()
        self.networks_data.gan_optimizer.zero_grad()
        rand_inputs = torch.rand(inputs_shape, device=self.state.device)

        gan_outputs = self.networks_data.gan(rand_inputs)
        self.networks_data.target_model.eval()

        target_model_outputs = self.networks_data.target_model(gan_outputs)
        self.networks_data.target_model.train()

        loss_gan = self.networks_data.gan_loss_function(rand_inputs.float(), gan_outputs, target_model_outputs)
        loss_gan.backward()

        self.state.loss_gan = loss_gan.item()
        if self.state.loss_gan < self.state.best_loss:
            self.saver.save_epoch(
                best_text="is_best",
                gan_outputs=gan_outputs,
                target_model_outputs=target_model_outputs,
                save_plot=True,
                epoch=self.state.epoch
            )

            self.state.best_loss = self.state.loss_gan

        torch.nn.utils.clip_grad_norm_(self.networks_data.gan.parameters(), 1.0)
        self.networks_data.gan_optimizer.step()

        return loss_gan

    def recover_from_nan(self, nan_recovery):
        nan_recovery.recover_from_nan_target_model()
        nan_recovery.recover_from_nan_gan()

    def is_gan_training_epoch(self):
        proportion_target_model_alone = torch.rand(1)
        return self.state.epoch >= self.hyperparameters.nr_steps_target_model_alone \
               and proportion_target_model_alone > self.hyperparameters.proportion_target_model_alone

    def train_models(self, model_index):
        for i, data in enumerate(self.data_loaders.train[model_index]):
            inputs, labels = data[0].to(self.state.device), data[1].to(self.state.device)
            inputs, labels = self.modifier((inputs, labels))

            if self.is_gan_training_epoch():
                for _ in range(self.hyperparameters.nr_steps_gan):
                    self.state.loss_gan = self.gan_train(list(inputs.shape))

                for _ in range(self.hyperparameters.nr_steps_target_model_gan):
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
                    f"Losses: target model = {loss_target_model:6.3f},  target model on gan = {loss_target_model_on_gan:6.3f}, "
                    f"gan = {self.state.loss_gan:6.3f}, "
                    f"Accuracies: target model = {accuracies_target_model:6.3f}"
                )

    def set_training_mode(self):
        self.networks_data.target_model.train()
        self.networks_data.gan.train()

    def train_model_with_index(self, model_index):
        nan_recovery = NetworkNaNRecovery(self.networks_data, self.paths.root_folder, self.state.device,                        self.data_loaders.train[model_index], self.modifier)
        
        for self.state.epoch in range(self.hyperparameters.total_epochs):
            start_time = time.time()
            self.recover_from_nan(nan_recovery)

            self.performances_logger.run(model_index)
            self.saver.save_best_validation_loss(performances=self.state.average_performances)
            self.state.best_loss = float("inf")

            self.set_training_mode()
            self.train_models(model_index)

            self.state.logger.log_execution_time(start_time, model_index)

        self.saver.save_model_data(model_index=model_index)

        if self.state.loss_gan != -1.0:
            self.networks_data.gan_scheduler.step()
        self.performances_logger.run(model_index)

    def initialize_data_for_new_model(self):
        self.state.epoch = 0
        self.networks_data = NetworksData(nr_dimensions=self.state.nr_dimensions,
                                          training_hyperparameters=self.hyperparameters,
                                          device=self.state.device,
                                          given_target_model=self.state.given_target_model,
                                          nr_classes=self.state.nr_classes)
        self.saver = Saver(device=self.state.device, modifier=self.modifier, networks_data=self.networks_data,
                           root_folder=self.paths.root_folder)
        self.performances_logger = PerformancesLogger(self)

        self.networks_data.gan = self.networks_data.gan.to(self.state.device)
        self.networks_data.load_models_if_present(path_to_load_target_model=self.paths.load_target_model,
                                                  path_to_load_gan=self.paths.load_gan)
        self.state.initialize_model_performances()

    def train(self):
        self.state.initialize_performances()
        self.state.logger.load_logs(path_to_performances=self.paths.performances,
                                    execution_data_file_name=self.execution_data_file_name,
                                    root_folder=self.paths.root_folder)

        for model_index in range(self.hyperparameters.k_fold):
            self.initialize_data_for_new_model()
            self.train_model_with_index(model_index)

        self.state.logger.log_execution_data(root_folder=self.paths.root_folder,
                                             execution_data_file_name=self.execution_data_file_name)
