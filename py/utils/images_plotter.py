import glob
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


class ImagesPlotter:
    def __init__(self, root_folder, target_model, modifier):
        self.root_folder = root_folder
        self.target_model = target_model
        self.modifier = modifier

    @staticmethod
    def create_gif(root_folder, pattern):
        filenames = np.sort(glob.glob(os.path.join(root_folder, "plots", pattern)))
        indexes = np.array(
            [
                el.replace(
                    os.path.join(
                        root_folder, "plots", pattern.replace("*", "").replace(".png", "")
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
            "{}/gifs/{}.gif".format(root_folder, pattern.replace("*", "").replace(".png", "")),
            images,
            loop=1,
            duration=0.01,
        )

    @torch.inference_mode()
    def plot_best_and_worst_examples(self, loader, epoch, device):
        self.target_model.eval()

        inputs, labels, _ = next(iter(loader))
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = self.modifier((inputs, labels))
        _, labels = torch.max(labels, 1)

        net_outputs = self.target_model(inputs)
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
            plot_title = self.get_plot_title(epoch=epoch, prediction=predicted[index].item(),
                                             prediction_score=score_prediction[index].item(), label=labels[index])
            self.plot_and_save_example(
                images=inputs[index].cpu().detach().numpy(),
                path_to_save="{}/plots/example-true-image-{}-step-{}.png".format(
                    self.root_folder, name, epoch
                ),
                plot_title=plot_title
            )

        self.target_model.train()

    @staticmethod
    def get_plot_title(epoch, prediction, prediction_score, label=-1):
        if label == -1:
            return "step = {}, prediction = {}, score = {}%".format(
                epoch, prediction, round(100.0 * np.nan_to_num(prediction_score))
            )
        return "step = {}, prediction = {}, score = {}%, truth = {}".format(
            epoch,
            prediction,
            round(100.0 * np.nan_to_num(prediction_score)),
            label,
        )

    @staticmethod
    def plot_and_save_example(images, path_to_save, plot_title):
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
        plt.title(plot_title)

        if legend:
            plt.legend(loc=1)
        plt.savefig(path_to_save)
        plt.clf()
