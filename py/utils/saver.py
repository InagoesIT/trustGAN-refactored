import os

import numpy as np
import torch

from py.utils.images_plotter import ImagesPlotter


class Saver:
    def __init__(self, networks_data, modifier, root_folder, device):
        self.networks_data = networks_data
        self.root_folder = root_folder
        self.modifier = modifier
        self.device = device

    def save_to_torch_full_model(self):
        checkpoint = {
            "model": self.networks_data.target_model,
            "state_dict": self.networks_data.target_model.state_dict(),
        }

        torch.save(checkpoint, "{}/target_model-fullModel.pth".format(self.root_folder))

    def save_model_data(self, epoch):
        if epoch % 50 == 0:
            torch.save(
                self.networks_data.target_model.state_dict(),
                "{}/nets/target_model-step-{}.pth".format(self.root_folder, epoch),
            )
            torch.save(self.networks_data.target_model.state_dict(),
                       "{}/target_model.pth".format(self.root_folder))
            torch.save(self.networks_data.gan.state_dict(), "{}/gan.pth".format(self.root_folder))
            self.save_to_torch_full_model()

    def save_best_validation_loss(self, performances):
        if (len(performances["valid"]["loss_net"]) == 1) or (
                performances["valid"]["loss_net"][-1]
                <= np.min(performances["valid"]["loss_net"][:-1])
        ):
            torch.save(
                self.networks_data.target_model.state_dict(),
                os.path.join(self.root_folder, "nets/target_model-is_best-valid-loss.pth"),
            )

    @torch.inference_mode()
    def save_epoch(
            self, epoch, best_text, gan_outputs=None, target_model_outputs=None, save_plot=True
    ):
        if save_plot:
            target_model_outputs = torch.nn.functional.softmax(target_model_outputs, dim=1)
            score_prediction, predicted = torch.max(target_model_outputs, 1)

            if score_prediction.ndim > 1:
                score_prediction = score_prediction.mean(axis=tuple(range(1, score_prediction.ndim)))
                predicted = predicted.to(torch.float).mean(
                    axis=tuple(range(1, predicted.ndim))
                )

            index = torch.argmax(score_prediction)
            plot_title = ImagesPlotter.get_plot_title(epoch=epoch, prediction=predicted[index].item(),
                                                      prediction_score=score_prediction[index].item())
            ImagesPlotter.plot_and_save_example(
                images=gan_outputs[index].cpu().detach().numpy(),
                path_to_save="{}/plots/example-image-{}-step-{}.png".format(
                    self.root_folder, best_text, epoch
                ),
                plot_title=plot_title
            )

        torch.save(
            self.networks_data.gan.state_dict(),
            "{}/nets/gan-{}-step-{}.pth".format(self.root_folder, best_text, epoch),
        )
