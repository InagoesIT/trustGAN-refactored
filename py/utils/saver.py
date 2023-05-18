import os

import numpy as np
import torch

from py.utils.plotter import Plotter


class Saver:
    def __init__(self, networks_data, modifier, plotter, root_folder, device):
        self.networks_data = networks_data
        self.root_folder = root_folder
        self.modifier = modifier
        self.device = device
        self.plotter = plotter

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
            self, epoch, best_text, loader=None, gan_outputs=None, net_outputs=None, save_plot=True
    ):
        if save_plot:
            if loader is not None:
                self.networks_data.target_model.eval()
                self.networks_data.gan.eval()
                dims = list(self.modifier(next(iter(loader)))[0].shape)
                rand_inputs = torch.rand(dims, device=self.device)

                gan_outputs = self.networks_data.gan(rand_inputs)
                net_outputs = self.networks_data.target_model(gan_outputs)
                self.networks_data.target_model.train()
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
            Plotter.plot_one_example(
                epoch=epoch,
                images=images,
                prediction=predicted[idx].item(),
                prediction_score=score_prediction[idx].item(),
                label=-1,
                path_to_save="{}/plots/example-image-{}-step-{}.png".format(
                    self.root_folder, best_text, epoch
                ),
            )

        torch.save(
            self.networks_data.gan.state_dict(),
            "{}/nets/gan-{}-step-{}.pth".format(self.root_folder, best_text, epoch),
        )
