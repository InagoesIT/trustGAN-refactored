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
import matplotlib.pyplot as plt
import glob
import numpy as np
import torch
from py.dataset.data_loader import DataLoader

from py.dataset.modifier import Modifier
from py.training.networks_data import NetworksData
from py.training.training_params import TrainingParams
from py.training.network_nan_recovery import NetworkNaNRecovery


class Training:
    # TODO: maybe needed to make this a builder?
    def __init__(
            self,
            training_params: TrainingParams,
            path_to_save,
            path2dataset,
            path_to_load_net=None,
            path_to_load_gan=None,
            path2net=None,
            device_name=None,
            verbose=True,
    ):
        self.data_loader_test = None
        self.data_loader_valid = None
        self.data_loader_train = None
        self.epoch = 0
        self.perfs = None
        self.best_loss = None
        self.verbose = verbose
        self.path_to_save = path_to_save
        self.path_to_load_net = path_to_load_net
        self.path_to_load_gan = path_to_load_gan
        self.training_params = training_params
        self.modifier = Modifier(nr_channels=self.training_params.nr_channels)

        self.process_path2save()
        self.data_loaders_train = []
        self.data_loaders_valid = []
        self.set_data_loaders(path2dataset)

        nr_dims = self.modifier(next(iter(self.data_loader_valid)))[0].ndim - 2
        print(f"INFO: Found {nr_dims} dimensions")
        self.training_params.nr_channels = self.modifier(next(iter(self.data_loader_valid)))[0].shape[1]
        print(f"INFO: Found {self.training_params.nr_channels} channels")

        self.device = self.get_device(device_name=device_name)
        self.sequence_dims_onnx = np.arange(2, 2 + nr_dims)

        self.networks_data = NetworksData(path2net, nr_dims, self.training_params)

        self.load_models_if_present()
        
        self.networks_data.gan = self.networks_data.gan.to(self.device)

    def set_data_loaders(self, path2dataset):
        self.data_loader_train = DataLoader.get_dataloader(
            path2dataset=path2dataset,
            nr_classes=self.training_params.nr_classes,
            dataset_type="train",
            batch_size=self.training_params.batch_size,
            verbose=self.verbose,
        )
        self.data_loader_valid = DataLoader.get_dataloader(
            path2dataset=path2dataset,
            nr_classes=self.training_params.nr_classes,
            dataset_type="valid",
            batch_size=self.training_params.batch_size,
            verbose=self.verbose,
        )
        self.data_loader_test = DataLoader.get_dataloader(
            path2dataset=path2dataset,
            nr_classes=self.training_params.nr_classes,
            dataset_type="test",
            batch_size=self.training_params.batch_size,
            verbose=self.verbose,
        )     
    
    def set_data_loaders_10_fold(self, k_fold=10):
        total_size = len(dataset)
        fold_size = int(total_size / k_fold)
        
        for fold_index in range(k_fold):
            validation_start_index = fold_index * fold_size
            validation_end_index = validation_start_index + fold_size
            
            validation_indices = list(range(validation_start_index,validation_end_index))
            train_left_indices = list(range(0, validation_start_index))
            train_right_indices = list(range(validation_end_index, total_size))

            train_indices = train_left_indices + train_right_indices

            train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
            val_set = torch.utils.data.dataset.Subset(dataset, val_indices)

            self.data_loaders_train.append(
                        torch.utils.data.DataLoader(train_set, shuffle=True,
                                            nr_classes=self.training_params.nr_classes,
                                            batch_size=self.training_params.batch_size,
                                            verbose=self.verbose))
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=50,
                                          shuffle=True, num_workers=4)

    def load_models_if_present(self):
        for model, path_to_load in [
            (self.networks_data.net, self.path_to_load_net),
            (self.networks_data.gan, self.path_to_load_gan),
        ]:
            model = model.to(self.device)

            if path_to_load is not None:
                ld = torch.load(path_to_load, map_location=self.device)
                model.load_state_dict(ld)

            if self.verbose:
                model.eval()
                # x_rand = torch.rand(
                #     self.modifier(next(iter(self.data_loader_valid)))[0].shape,
                #     device=self.device,
                # )
                # torchsummaryX.summary(model, x_rand)
                model.train()

    def process_path2save(self):
        if self.path_to_save is not None:
            if not os.path.isdir(self.path_to_save):
                os.mkdir(self.path_to_save)
            for folder in ["plots", "nets", "perfs-plots", "gifs"]:
                if not os.path.isdir(os.path.join(self.path_to_save, folder)):
                    os.mkdir(os.path.join(self.path_to_save, folder))
                else:
                    print("\nWARNING\n Files exists")

    def get_device(self, device_name=None):
        if device_name is None:
            device_name = "cuda:0"

        if not torch.cuda.is_available():
            device_name = "cpu"

        device = torch.device(device_name)
        if self.verbose:
            print(f"Device = {device}, {device_name}")

        return device
        

    @torch.inference_mode()
    def get_predictions(self, loader, score_type="MCP"):
        if type(loader) == str:
            loader = getattr(self, loader)

        self.networks_data.net.eval()

        softmax = torch.nn.Softmax(dim=1)
        truth = []
        predictions = []
        score = []

        for data in loader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs, labels = self.modifier((inputs, labels))

            outputs = softmax(self.networks_data.net(inputs))
            _, tmp_predictions = torch.max(outputs, 1)
            outputs, _ = torch.sort(outputs, dim=1)
            if score_type == "MCP":
                tmp_score = outputs[:, -1]
            elif score_type == "Diff2MCP":
                tmp_score = outputs[:, -1] - outputs[:, -2]
            _, tmp_truth = torch.max(labels, 1)

            truth += [tmp_truth.detach().cpu().numpy()]
            predictions += [tmp_predictions.detach().cpu().numpy()]
            score += [tmp_score.detach().cpu().numpy()]

        truth = np.hstack(truth)
        predictions = np.hstack(predictions)
        score = np.hstack(score)

        self.networks_data.net.train()

        return truth, predictions, score

    # performance metrics
    @torch.inference_mode()
    def get_perfs(self, loader, header_str=""):
        self.networks_data.net.eval()
        self.networks_data.gan.eval()

        accuracies = {"net": 0.0, "net_on_gan": 0.0, "gan": 0.0}
        loss = {"net": 0.0, "net_on_gan": 0.0, "gan": 0.0}

        for data in loader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs, labels = self.modifier((inputs, labels))

            # Net on real data
            outputs = self.networks_data.net(inputs)
            loss["net"] += (
                self.networks_data.net_loss(outputs, labels, reduction="sum").detach().cpu().numpy()
            )
            _, hard_predicted = torch.max(outputs, 1)
            _, hard_labels = torch.max(labels, 1)
            accuracies["net"] += (
                (hard_predicted == hard_labels).float().sum().detach().cpu().numpy()
            )

            # Net on Gan generated images and gan loss
            rand_inputs = torch.rand(inputs.shape, device=self.device)
            rand_labels = (
                    1.0 / self.training_params.nr_classes * torch.ones(labels.shape, device=self.device)
            )

            gan_outputs = self.networks_data.gan(rand_inputs)
            net_outputs = self.networks_data.net(gan_outputs)

            loss["net_on_gan"] += (
                self.networks_data.net_loss(net_outputs, rand_labels, reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )
            loss["gan"] += (
                self.networks_data.gan_loss(rand_inputs, gan_outputs, net_outputs, reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )

        for k in accuracies.keys():
            accuracies[k] = accuracies[k] / loader.dataset.nr_total_labels
        for k in loss.keys():
            loss[k] = loss[k] / loader.dataset.nr_total_labels

        self.networks_data.net.train()
        self.networks_data.gan.train()

        res_str = header_str + ": Losses: "
        for k, v in loss.items():
            res_str += f"{k} = {v:6.3f}, "
        res_str += "Accuracy: "
        for k, v in accuracies.items():
            res_str += f"{k} = {v:6.3f}, "

        print(res_str)

        return accuracies, loss

    def plot_one_example(self, x, prediction, score_prediction, label, path2save):
        """
        Plot and save one example
        """

        if len(x.shape) == 3:
            x = (x + 1.0) / 2.0
            legend = False
            if x.shape[0] == 1:
                x = x[0, :, :]
                plt.imshow(x, cmap="gray")
            elif x.shape[0] == 3:
                x = np.concatenate(
                    [x[0][:, :, None], x[1][:, :, None], x[2][:, :, None]], axis=-1
                )
                plt.imshow(x)
            else:
                raise ValueError(
                    "ERROR: x does not have the proper dimension: {}".format(x.shape)
                )
        elif len(x.shape) == 2:
            legend = True
            for i in range(x.shape[0]):
                plt.plot(x[i, :], label=r"dim={}".format(i))
        else:
            raise ValueError(
                "ERROR: x does not have the proper dimension: {}".format(x.shape)
            )

        if label == -1:
            plt.title(
                "step = {}, prediction = {}, score = {}%".format(
                    self.epoch, prediction, round(100.0 * np.nan_to_num(score_prediction))
                )
            )
        else:
            plt.title(
                "step = {}, prediction = {}, score = {}%, truth = {}".format(
                    self.epoch,
                    prediction,
                    round(100.0 * np.nan_to_num(score_prediction)),
                    label,
                )
            )

        if legend:
            plt.legend(loc=1)
        plt.savefig(path2save)
        plt.clf()

    @torch.inference_mode()
    def save_epoch(
            self, best, loader=None, gan_outputs=None, net_outputs=None, save_plot=True
    ):
        """ """

        if save_plot:
            if loader is not None:
                self.networks_data.net.eval()
                self.networks_data.gan.eval()
                dims = list(self.modifier(next(iter(loader)))[0].shape)
                rand_inputs = torch.rand(dims, device=self.device)

                gan_outputs = self.networks_data.gan(rand_inputs)
                net_outputs = self.networks_data.net(gan_outputs)
                self.networks_data.net.train()
                self.networks_data.gan.train()

            net_outputs = torch.nn.functional.softmax(net_outputs, dim=1)
            score_prediction, predicted = torch.max(net_outputs, 1)

            if score_prediction.ndim > 1:
                score_prediction = score_prediction.mean(axis=tuple(range(1, score_prediction.ndim)))
                predicted = predicted.to(torch.float).mean(
                    axis=tuple(range(1, predicted.ndim))
                )

            idx = torch.argmax(score_prediction)

            images = gan_outputs[idx].cpu().detach().numpy()
            self.plot_one_example(
                images,
                prediction=predicted[idx].item(),
                score_prediction=score_prediction[idx].item(),
                label=-1,
                path2save="{}/plots/example-image-{}-step-{}.png".format(
                    self.path_to_save, best, self.epoch
                ),
            )

        torch.save(
            self.networks_data.gan.state_dict(),
            "{}/nets/gan-{}-step-{}.pth".format(self.path_to_save, best, self.epoch),
        )

    @torch.inference_mode()
    def get_example(self, loader):
        """ """

        self.networks_data.net.eval()

        inputs, labels = next(iter(loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        inputs, labels = self.modifier((inputs, labels))
        _, labels = torch.max(labels, 1)

        net_outputs = self.networks_data.net(inputs)

        net_outputs = torch.nn.functional.softmax(net_outputs, dim=1)
        score_pred, predicted = torch.max(net_outputs, 1)

        if score_pred.ndim > 1:
            score_pred = score_pred.mean(axis=tuple(range(1, score_pred.ndim)))
            predicted = predicted.to(torch.float).mean(
                axis=tuple(range(1, predicted.ndim))
            )
            labels = labels.to(torch.float).mean(axis=tuple(range(1, labels.ndim)))

        idx_min = torch.argmin(score_pred)
        idx_max = torch.argmax(score_pred)

        for idx, name in [(idx_min, "min"), (idx_max, "max")]:
            images = inputs[idx].cpu().detach().numpy()

            self.plot_one_example(
                images,
                prediction=predicted[idx].item(),
                score_prediction=score_pred[idx].item(),
                label=labels[idx],
                path2save="{}/plots/example-true-image-{}-step-{}.png".format(
                    self.path_to_save, name, self.epoch
                ),
            )

        self.networks_data.net.train()

    def log_perfs(self):
        """ """

        for dataset, name in [(self.data_loader_train, "train"), (self.data_loader_valid, "valid")]:
            accuracies, loss = self.get_perfs(
                loader=dataset, header_str="{} {}".format(self.epoch, name)
            )

            for met, met_name in [(accuracies, "accs"), (loss, "loss")]:

                for k, v in met.items():
                    final_name = "{}_{}".format(met_name, k)
                    if final_name not in self.perfs[name].keys():
                        self.perfs[name][final_name] = []

                    self.perfs[name][final_name] += [v]

    def net_train(self, inputs, labels):
        """ """

        self.networks_data.net.train()
        self.networks_data.net_optim.zero_grad()

        net_outputs = self.networks_data.net(inputs)

        loss_net = self.networks_data.net_loss(net_outputs, labels)
        loss_net.backward()

        torch.nn.utils.clip_grad_norm_(self.networks_data.net.parameters(), self.networks_data.grad_clipping_coeff)
        self.networks_data.net_optim.step()

        _, truth = torch.max(labels, 1)
        _, predicted = torch.max(net_outputs, 1)
        acc_net = (predicted == truth).float().mean()

        return loss_net, acc_net

    def net_gan_train(self, inputs_shape, labels_shape):
        """ """

        self.networks_data.net.train()
        self.networks_data.net_optim.zero_grad()

        rand_inputs = torch.rand(inputs_shape, device=self.device)
        rand_labels = (
                1.0 / self.training_params.nr_classes * torch.ones(labels_shape, device=self.device)
        )

        explore_probability = torch.rand(1)
        nets = glob.glob("{}/nets/gan-*-step-*.pth".format(self.path_to_save))

        # load a previous gan model or use the current one
        gan_to_use = self.networks_data.gan
        if (explore_probability < 0.1) and (len(nets) > 0):
            nr_network = torch.randint(low=0, high=len(nets), size=[1])
            self.networks_data.gan2.load_state_dict(torch.load(nets[nr_network]))
            gan_to_use = self.networks_data.gan2

        # get the generated output of the gan (don't train it!)
        gan_to_use.eval()
        gan_outputs = gan_to_use(rand_inputs)
        gan_to_use.train()

        # backprop the loss
        net_outputs = self.networks_data.net(gan_outputs)
        loss_net_gan = self.networks_data.net_loss(net_outputs, rand_labels)
        loss_net_gan.backward()
        torch.nn.utils.clip_grad_norm_(self.networks_data.net.parameters(), self.networks_data.grad_clipping_coeff)
        self.networks_data.net_optim.step()

        return loss_net_gan

    def gan_train(self, inputs_shape):
        self.networks_data.gan.train()
        self.networks_data.gan_optim.zero_grad()
        rand_inputs = torch.rand(inputs_shape, device=self.device)

        gan_outputs = self.networks_data.gan(rand_inputs)
        self.networks_data.net.eval()

        net_outputs = self.networks_data.net(gan_outputs)
        self.networks_data.net.train()

        loss_gan = self.networks_data.gan_loss(rand_inputs.float(), gan_outputs, net_outputs)
        loss_gan.backward()

        loss_gan = loss_gan.item()
        if loss_gan < self.best_loss:
            self.save_epoch(
                best="best",
                gan_outputs=gan_outputs,
                net_outputs=net_outputs,
                save_plot=True,
            )

            self.best_loss = loss_gan

        torch.nn.utils.clip_grad_norm_(self.networks_data.gan.parameters(), 1.0)
        self.networks_data.gan_optim.step()

        return loss_gan

    def train(self):
        self.perfs = {"train": {}, "valid": {}}
        self.perfs["train"]["best-gan-loss"] = []
        self.perfs["valid"]["best-gan-loss"] = []
        self.best_loss = float("inf")
        loss_gan = -1.0
        
        # self.perfs = np.load("{}/performances.npy".format(self.path_to_save), allow_pickle=True)
        # self.perfs = self.perfs.item()
        
        for self.epoch in range(self.training_params.nr_epochs):
            nan_recovery = NetworkNaNRecovery(self.networks_data, self.path_to_save, self.device, self.data_loader_train, self.modifier)
            nan_recovery.recover_from_nan_net()
            nan_recovery.recover_from_nan_gan()
            self.perfs["train"]["best-gan-loss"] += [self.best_loss]
            self.perfs["valid"]["best-gan-loss"] += [-1.0]
            self.log_perfs()
            self.get_example(loader=self.data_loader_train)
            self.save_epoch(best="not-best", loader=self.data_loader_train)
            np.save("{}/performances.npy".format(self.path_to_save), self.perfs)

            if (len(self.perfs["valid"]["loss_net"]) == 1) or (
                    self.perfs["valid"]["loss_net"][-1]
                    <= np.min(self.perfs["valid"]["loss_net"][:-1])
            ):
                torch.save(
                    self.networks_data.net.state_dict(),
                    os.path.join(self.path_to_save, "nets/net-best-valid-loss.pth"),
                )

            self.best_loss = float("inf")

            # set training mode
            self.networks_data.net.train()
            self.networks_data.gan.train()

            # Train the classifier
            for i, data in enumerate(self.data_loader_train):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                inputs, labels = self.modifier((inputs, labels))

                proportion_net_alone = torch.rand(1)
                if (self.epoch >= self.training_params.nr_step_net_alone) and \
                        (proportion_net_alone > self.training_params.proportion_net_alone):
                    for _ in range(self.training_params.nr_step_gan):
                        loss_gan = self.gan_train(list(inputs.shape))

                    for _ in range(self.training_params.nr_step_net_gan):
                        loss_net_gan = self.net_gan_train(
                            list(inputs.shape), list(labels.shape)
                        )
                else:
                    loss_net_gan = -1.0
                    loss_gan = -1.0

                loss_net, acc_net = self.net_train(inputs, labels)

                if i % 10 == 0:
                    print(
                        f"{i:03d}/{len(self.data_loader_train):03d}, Loss: net = {loss_net:6.3f}, net_on_gan = {loss_net_gan:6.3f}, gan = {loss_gan:6.3f}, Accs: net = {acc_net:6.3f}"
                    )

            if self.epoch % 100 == 0:
                torch.save(
                    self.networks_data.net.state_dict(),
                    "{}/nets/net-step-{}.pth".format(self.path_to_save, self.epoch),
                )
            torch.save(self.networks_data.net.state_dict(), "{}/net.pth".format(self.path_to_save))
            torch.save(self.networks_data.gan.state_dict(), "{}/gan.pth".format(self.path_to_save))
            self.save_to_torch_full_model()

        if loss_gan != -1.0:
            self.networks_data.gan_scheduler.step()

        self.epoch = self.training_params.nr_epochs
        self.perfs["train"]["best-gan-loss"] += [self.best_loss]
        self.perfs["valid"]["best-gan-loss"] += [-1.0]
        self.log_perfs()
        self.get_example(loader=self.data_loader_train)
        self.save_epoch(best="not-best", loader=self.data_loader_train)
        np.save("{}/performances.npy".format(self.path_to_save), self.perfs)

    def save_to_torch_full_model(self):
        """
        Save to Torch full model
        """

        checkpoint = {
            "model": self.networks_data.net,
            "state_dict": self.networks_data.net.state_dict(),
        }

        torch.save(checkpoint, "{}/net-fullModel.pth".format(self.path_to_save))
