import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from py.utils.images_plotter import ImagesPlotter


class PerformancesLogger:
    base_name_for_model_performances_file = "model_performances_"

    def __init__(self, training):
        self.training = training
        self.images_plotter = ImagesPlotter(root_folder=training.paths.root_folder,
                                            target_model=training.networks_data.target_model,
                                            modifier=training.modifier)
        self.writer_path = f"{self.training.paths.root_folder}/tensorboard/{self.training.state.model_label}"
        self.writer = SummaryWriter(self.writer_path)

    @staticmethod
    def get_validation_epochs(epochs, validation_at, total_epochs):
        epochs = [epoch for epoch in epochs if
                  PerformancesLogger.is_validation_epoch(epoch=epoch,
                                                         validation_at=validation_at,
                                                         is_last_epoch=epoch == total_epochs)]
        return epochs

    @torch.inference_mode()
    def get_performances(self, loader, header_str=""):
        self.training.networks_data.target_model.eval()
        self.training.networks_data.gan.eval()

        accuracies = {"target_model": 0.0, "target_model_on_gan": 0.0, "gan": 0.0}
        loss = {"target_model": 0.0, "target_model_on_gan": 0.0, "gan": 0.0}

        for data in loader:
            inputs, labels = data[0].to(self.training.state.device), data[1].to(self.training.state.device)
            inputs, labels = self.training.modifier((inputs, labels))

            # Net on real state
            outputs = self.training.networks_data.target_model(inputs)
            loss["target_model"] += (
                self.training.networks_data.target_model_loss_function(outputs, labels,
                                                                       reduction="sum").detach().cpu().numpy()
            )
            _, hard_predicted = torch.max(outputs, 1)
            _, hard_labels = torch.max(labels, 1)
            accuracies["target_model"] += (
                (hard_predicted == hard_labels).float().sum().detach().cpu().numpy()
            )

            # Net on Gan generated images and gan loss
            rand_inputs = torch.rand(inputs.shape, device=self.training.state.device)
            rand_labels = (
                    1.0 / self.training.hyperparameters.nr_classes *
                    torch.ones(labels.shape, device=self.training.state.device)
            )

            gan_outputs = self.training.networks_data.gan(rand_inputs)
            target_model_outputs = self.training.networks_data.target_model(gan_outputs)
            loss["target_model_on_gan"] += (
                self.training.networks_data.target_model_on_gan_loss_function(target_model_outputs, rand_labels, reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )
            loss["gan"] += (
                self.training.networks_data.gan_loss_type(rand_inputs, gan_outputs, target_model_outputs,
                                                          reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )

        for k in accuracies.keys():
            accuracies[k] = accuracies[k] / loader.dataset.nr_total_labels
        for k in loss.keys():
            loss[k] = loss[k] / loader.dataset.nr_total_labels

        self.training.networks_data.target_model.train()
        self.training.networks_data.gan.train()

        res_str = header_str + ": Losses: "
        for k, v in loss.items():
            res_str += f"{k} = {v:6.3f}, "
        res_str += "Accuracy: "
        for k, v in accuracies.items():
            res_str += f"{k} = {v:6.3f}, "

        print(res_str)

        return accuracies, loss

    @staticmethod
    def is_validation_epoch(epoch, validation_at, is_last_epoch=False):
        return (epoch + 1) % validation_at == 0 or epoch == 0 or is_last_epoch

    def add_data_to_model_performances(self, metric_final_name, dataset_type, value):
        if metric_final_name not in self.training.state.model_performances[dataset_type].keys():
            self.training.state.model_performances[dataset_type][metric_final_name] = []
        self.training.state.model_performances[dataset_type][metric_final_name] += [value]

    def add_data_to_average_performances(self, metric_final_name, dataset_type, value, epoch, model_index):
        if metric_final_name not in self.training.state.average_performances[dataset_type].keys():
            self.training.state.average_performances[dataset_type][metric_final_name] = []
        if model_index == 0:
            self.training.state.average_performances[dataset_type][metric_final_name] += [value]
            return
        self.training.state.average_performances[dataset_type][metric_final_name][epoch] += value
        self.training.state.average_performances[dataset_type][metric_final_name][epoch] /= 2

    def is_one_epoch_and_wants_last_performances(self):
        return self.training.hyperparameters.total_epochs == 1 and \
            len(self.training.state.model_performances["training"]) > 0

    def average_performances_were_calculated(self):
        return self.training.state.epoch == 0 and \
            self.training.paths.path_to_performances is not None

    def set_performances_for_dataset(self, accuracies, losses, model_index, dataset_type, epoch, add_to_average):
        for metric, metric_name in [(accuracies, "accuracy"), (losses, "loss")]:
            for network_task, value in metric.items():
                metric_final_name = "{}_{}".format(metric_name, network_task)
                if add_to_average:
                    self.add_data_to_average_performances(metric_final_name, dataset_type, value, epoch, model_index)
                self.add_data_to_model_performances(metric_final_name, dataset_type, value)

    def calculate_performances(self, model_index):
        dataloaders_and_dataset_types = [(self.training.data_loaders.train[model_index], "training")]
        epoch = self.training.state.epoch
        if PerformancesLogger.is_validation_epoch(
                epoch=epoch, validation_at=self.training.hyperparameters.validation_interval,
                is_last_epoch=epoch == self.training.hyperparameters.total_epochs - 1):
            dataloaders_and_dataset_types.append((self.training.data_loaders.validation[model_index], "validation"))

        add_to_average = self.is_one_epoch_and_wants_last_performances() or \
                         not self.average_performances_were_calculated()
        for dataloader, dataset_type in dataloaders_and_dataset_types:
            accuracies, losses = self.get_performances(
                loader=dataloader, header_str="{} {}".format(self.training.state.epoch, dataset_type)
            )
            if dataset_type == "validation":
                epoch = (epoch + 1) // self.training.hyperparameters.validation_interval

            self.set_performances_for_dataset(accuracies, losses, model_index, dataset_type, epoch, add_to_average)

    def run(self, model_index):
        self.calculate_performances(model_index)
        self.images_plotter.plot_best_and_worst_examples(loader=self.training.data_loaders.train[model_index],
                                                         epoch=self.training.state.epoch,
                                                         device=self.training.state.device)
        self.training.saver.save_epoch(best_text="not-best", epoch=self.training.state.epoch,
                                       loader=self.training.data_loaders.train[model_index])
        np.save("{}/average_performances.npy".format(self.training.paths.root_folder),
                self.training.state.average_performances)
        np.save("{}/{}{}.npy".format(self.training.paths.root_folder,
                                     PerformancesLogger.base_name_for_model_performances_file,
                                     model_index),
                self.training.state.model_performances)
