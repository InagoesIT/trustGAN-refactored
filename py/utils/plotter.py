import glob
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from py.performances.performances_logger import PerformancesLogger


class Plotter:
    def __init__(self, root_folder, device, networks_data, modifier, total_epochs, validation_interval):
        self.root_folder = root_folder
        self.networks_data = networks_data
        self.modifier = modifier
        self.device = device
        self.total_epochs = total_epochs
        self.validation_interval = validation_interval

    def plot_performances(self):
        performances = np.load("{}/performances.npy".format(self.root_folder), allow_pickle=True)
        performances = performances.item()

        train_metrics = list(performances[list(performances.keys())[0]].keys())

        for metric in train_metrics:
            for dataset_type in performances.keys():
                performances = performances[dataset_type][metric]
                epochs = [epoch for epoch in range(self.total_epochs)]
                if performances.length != self.total_epochs:
                    epochs = [epoch for epoch in epochs if
                              PerformancesLogger.is_validation_epoch(epoch=epoch,
                                                                     validation_at=self.validation_interval)]
                plt.plot(epochs, performances, label=dataset_type)

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid()
            plt.savefig("{}/performance-plots/{}.png".format(self.root_folder, metric))
            plt.clf()

    def plot_execution_time(self):
        time_data = np.load("{}/execution_data.npy".format(self.root_folder), allow_pickle=True)
        time_data = time_data.item()

        for epoch, time in enumerate(time_data["time"]):
            plt.plot(epoch, time)
        plt.xlabel("Epoch")
        plt.ylabel("Execution time in minutes")
        plt.grid()
        plt.savefig("{}/time-plot.png".format(self.root_folder))
        plt.clf()

    def create_gif(self, pattern):
        filenames = np.sort(glob.glob(os.path.join(self.root_folder, "plots", pattern)))
        indexes = np.array(
            [
                el.replace(
                    os.path.join(
                        self.root_folder, "plots", pattern.replace("*", "").replace(".png", "")
                    ),
                    "",
                ).replace(".png", "")
                for el in filenames
            ]
        ).astype(int)
        sorted_indexes = np.argsort(indexes)
        filenames = filenames[sorted_indexes]

        print(f"Create gifs, {len(filenames)} files found for {pattern}")
        if len(filenames) == 0:
            return

        images = [imageio.imread(filename) for filename in filenames]

        imageio.mimsave(
            "{}/gifs/{}.gif".format(self.root_folder, pattern.replace("*", "").replace(".png", "")),
            images,
            loop=1,
            duration=0.01,
        )

    @torch.inference_mode()
    def plot_best_and_worst_examples(self, loader, epoch):
        self.networks_data.target_model.eval()

        inputs, labels = next(iter(loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        inputs, labels = self.modifier((inputs, labels))
        _, labels = torch.max(labels, 1)

        net_outputs = self.networks_data.target_model(inputs)
        net_outputs = torch.nn.functional.softmax(net_outputs, dim=1)
        score_prediction, predicted = torch.max(net_outputs, 1)

        if score_prediction.ndim > 1:
            score_prediction = score_prediction.mean(axis=tuple(range(1, score_prediction.ndim)))
            predicted = predicted.to(torch.float).mean(
                axis=tuple(range(1, predicted.ndim))
            )
            labels = labels.to(torch.float).mean(axis=tuple(range(1, labels.ndim)))

        min_index = torch.argmin(score_prediction)
        max_index = torch.argmax(score_prediction)
        for index, name in [(min_index, "min"), (max_index, "max")]:
            images = inputs[index].cpu().detach().numpy()

            self.plot_one_example(
                epoch=epoch,
                images=images,
                prediction=predicted[index].item(),
                prediction_score=score_prediction[index].item(),
                label=labels[index],
                path_to_save="{}/plots/example-true-image-{}-step-{}.png".format(
                    self.root_folder, name, epoch
                ),
            )

        self.networks_data.target_model.train()

    @staticmethod
    def plot_one_example(epoch, images, prediction, prediction_score, label, path_to_save):
        """
        Plot and save one example
        """

        if len(images.shape) == 3:
            images = (images + 1.0) / 2.0
            legend = False
            if images.shape[0] == 1:
                images = images[0, :, :]
                plt.imshow(images, cmap="gray")
            elif images.shape[0] == 3:
                images = np.concatenate(
                    [images[0][:, :, None], images[1][:, :, None], images[2][:, :, None]], axis=-1
                )
                plt.imshow(images)
            else:
                raise ValueError(
                    "ERROR: images does not have the proper dimension: {}".format(images.shape)
                )
        elif len(images.shape) == 2:
            legend = True
            for i in range(images.shape[0]):
                plt.plot(images[i, :], label=r"dim={}".format(i))
        else:
            raise ValueError(
                "ERROR: images does not have the proper dimension: {}".format(images.shape)
            )

        if label == -1:
            plt.title(
                "step = {}, prediction = {}, score = {}%".format(
                    epoch, prediction, round(100.0 * np.nan_to_num(prediction_score))
                )
            )
        else:
            plt.title(
                "step = {}, prediction = {}, score = {}%, truth = {}".format(
                    epoch,
                    prediction,
                    round(100.0 * np.nan_to_num(prediction_score)),
                    label,
                )
            )

        if legend:
            plt.legend(loc=1)
        plt.savefig(path_to_save)
        plt.clf()

