import pkgutil

import torch
import copy

import torchmetrics as torchmetrics

from py.networks.gan import Gan
from py.performances.losses import Losses

package = __import__("py.networks")


class NetworksData:
    def __init__(self, nr_dimensions, training_hyperparameters, device, given_target_model=None):
        self.training_hyperparameters = training_hyperparameters
        self.nr_dimensions = nr_dimensions
        self.device = device

        self.target_model = None
        self.set_target_model(given_target_model)
        self.target_model_loss_function = self.get_loss_function_for(
            self.training_hyperparameters.target_model_loss)
        self.target_model_on_gan_loss_function = self.get_loss_function_for(
            self.training_hyperparameters.target_model_on_gan_loss)
        self.target_model_optimizer = torch.optim.AdamW(self.target_model.parameters(), weight_decay=0.05)

        self.gan = None
        self.set_gan()
        self.gan_loss_type = Losses.get_combined_gan_loss
        self.gan_optimizer = torch.optim.AdamW(self.gan.parameters(), weight_decay=0.05)
        self.gan_scheduler = None
        self.set_gan_scheduler()
        self.gan_copy = copy.deepcopy(self.gan)

        self.recovered_from_nan_target_model = 0
        self.recovered_from_nan_gan = 0
        self.grad_clipping_coefficient = 1.0

    def set_target_model(self, given_target_model):
        if given_target_model is not None:
            self.target_model = given_target_model
            return
        for importer, modname, is_pkg in pkgutil.walk_packages(package.__path__):
            module = importer.find_module(modname).load_module(modname)

            if not hasattr(module, self.training_hyperparameters.target_model_network_type):
                continue
            network = getattr(module, self.training_hyperparameters.target_model_network_type)
            if self.training_hyperparameters.target_model_network_type == 'Net':
                self.target_model = network(
                    nr_classes=self.training_hyperparameters.nr_classes,
                    nr_channels=self.training_hyperparameters.nr_channels,
                    is_batch_norm=False,
                    is_weight_norm=True,
                    dim=f"{self.nr_dimensions}d",
                    residual_units_number=self.training_hyperparameters.target_model_residual_units_number
                )
            else:
                self.target_model = network(
                    nr_classes=self.training_hyperparameters.nr_classes,
                    nr_channels=self.training_hyperparameters.nr_channels,
                    is_batch_norm=False,
                    is_weight_norm=True,
                    dim=f"{self.nr_dimensions}d",
                )
            break

    def get_loss_function_for(self, loss_name):
        loss = Losses.get_softmax_cross_entropy_loss
        if loss_name == 'hinge':
            loss = torchmetrics.classification.MulticlassHingeLoss(
                num_classes=self.training_hyperparameters.nr_classes).to(self.device)
        elif loss_name == 'squared hinge':
            loss = torchmetrics.classification.MulticlassHingeLoss(
                num_classes=self.training_hyperparameters.nr_classes, squared=True).to(self.device)
        elif loss_name == 'cubed hinge':
            loss = Losses.get_cubed_hinge_loss
        elif loss_name == 'cauchy-schwarz':
            loss = Losses.get_cauchy_schwarz_divergence
        return loss

    def set_gan(self):
        self.gan = Gan(
            nr_channels=self.training_hyperparameters.nr_channels,
            is_batch_norm=True,
            is_weight_norm=False,
            dim=f"{self.nr_dimensions}d",
            residual_units_number=self.training_hyperparameters.gan_residual_units_number
        )

    def set_gan_scheduler(self):
        self.gan_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.gan_optimizer,
            base_lr=1.0e-3,
            max_lr=5.0e-3,
            step_size_up=50,
            cycle_momentum=False,
        )
