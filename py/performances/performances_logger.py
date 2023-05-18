import numpy as np
import torch


class PerformancesLogger:
    def __init__(self, training):
        self.training = training

    @torch.inference_mode()
    def get_performances(self, loader, header_str=""):
        self.training.models_data.target_model.eval()
        self.training.models_data.gan.eval()

        accuracies = {"target_model": 0.0, "net_on_gan": 0.0, "gan": 0.0}
        loss = {"target_model": 0.0, "net_on_gan": 0.0, "gan": 0.0}

        for data in loader:
            inputs, labels = data[0].to(self.training.state.device), data[1].to(self.training.state.device)
            inputs, labels = self.training.modifier((inputs, labels))

            # Net on real state
            outputs = self.training.models_data.target_model(inputs)
            loss["target_model"] += (
                self.training.models_data.net_loss(outputs, labels, reduction="sum").detach().cpu().numpy()
            )
            _, hard_predicted = torch.max(outputs, 1)
            _, hard_labels = torch.max(labels, 1)
            accuracies["target_model"] += (
                (hard_predicted == hard_labels).float().sum().detach().cpu().numpy()
            )

            # Net on Gan generated images and gan loss
            rand_inputs = torch.rand(inputs.shape, device=self.training.state.device)
            rand_labels = (
                    1.0 / self.training.parameters.nr_classes *
                    torch.ones(labels.shape, device=self.training.state.device)
            )

            gan_outputs = self.training.models_data.gan(rand_inputs)
            net_outputs = self.training.models_data.target_model(gan_outputs)

            loss["net_on_gan"] += (
                self.training.models_data.net_loss(net_outputs, rand_labels, reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )
            loss["gan"] += (
                self.training.models_data.gan_loss(rand_inputs, gan_outputs, net_outputs, reduction="sum")
                .detach()
                .cpu()
                .numpy()
            )

        for k in accuracies.keys():
            accuracies[k] = accuracies[k] / loader.dataset.nr_total_labels
        for k in loss.keys():
            loss[k] = loss[k] / loader.dataset.nr_total_labels

        self.training.models_data.target_model.train()
        self.training.models_data.gan.train()

        res_str = header_str + ": Losses: "
        for k, v in loss.items():
            res_str += f"{k} = {v:6.3f}, "
        res_str += "Accuracy: "
        for k, v in accuracies.items():
            res_str += f"{k} = {v:6.3f}, "

        print(res_str)

        return accuracies, loss

    @staticmethod
    def is_validation_epoch(epoch, validation_at):
        return (epoch + 1) % validation_at == 0 or epoch == 0

    def calculate_performances(self, model_index):
        dataloaders_and_dataset_types = [(self.training.data_loaders.train[model_index], "train")]
        epoch = self.training.training.state.epoch
        if PerformancesLogger.is_validation_epoch(
                epoch=epoch, validation_at=self.training.parameters.validation_interval):
            dataloaders_and_dataset_types.append((self.training.data_loaders.validation[model_index], "valid"))

        for dataloader, dataset_type in dataloaders_and_dataset_types:
            accuracies, losses = self.get_performances(
                loader=dataloader, header_str="{} {}".format(self.training.state.epoch, dataset_type)
            )
            if dataset_type == "valid":
                epoch = (epoch + 1) // self.training.state.validation_interval

            self.set_performances_for_dataset(accuracies, losses, model_index, dataset_type, epoch)

    def set_performances_for_dataset(self, accuracies, losses, model_index, dataset_type, epoch):
        for metric, metric_name in [(accuracies, "accs"), (losses, "loss")]:
            for network_task, value in metric.items():
                metric_final_name = "{}_{}".format(metric_name, network_task)
                if metric_final_name not in self.training.state.perfs[dataset_type].keys():
                    self.training.state.perfs[dataset_type][metric_final_name] = []
                if model_index == 0:
                    self.training.state.perfs[dataset_type][metric_final_name] += [value]
                    continue
                self.training.state.perfs[dataset_type][metric_final_name][epoch] += value
                self.training.state.perfs[dataset_type][metric_final_name][epoch] /= 2

    def run(self, model_index):
        if model_index == 0:
            self.training.state.perfs["train"]["is_best-gan-loss"] += [self.training.state.best_loss]
            self.training.state.perfs["valid"]["is_best-gan-loss"] += [-1.0]
        else:
            self.training.state.perfs["train"]["is_best-gan-loss"][self.training.state.epoch] += self.training.state.best_loss
            self.training.state.perfs["train"]["is_best-gan-loss"][self.training.state.epoch] /= 2
            self.training.state.perfs["valid"]["is_best-gan-loss"][self.training.state.epoch] += -1.0
            self.training.state.perfs["valid"]["is_best-gan-loss"][self.training.state.epoch] /= 2

        self.calculate_performances(model_index)
        self.training.plotter.plot_best_and_worst_examples(loader=self.training.data_loaders.train[model_index])
        self.training.saver.save_epoch(best="not-is_best", loader=self.training.data_loaders.train[model_index])
        np.save("{}/performances.npy".format(self.training.paths.root_folder), self.training.state.perfs)
