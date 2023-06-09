import pkgutil

import torch
import copy

from py.networks.gan import Gan
from py.performances.losses import Losses

package = __import__("py.networks")


class NetworksData:
    def __init__(self, nr_dimensions, nr_classes, training_hyperparameters, device, given_target_model=None):
        self.training_hyperparameters = training_hyperparameters
        self.nr_dimensions = nr_dimensions
        self.device = device
        self.nr_classes = nr_classes

        self.target_model = None
        self.set_target_model(given_target_model)
        self.target_model_loss_function = Losses.get_loss_function_for(
            loss_name=self.training_hyperparameters.target_model_loss)
        self.target_model_on_gan_loss_function = Losses.get_loss_function_for(
            self.training_hyperparameters.target_model_on_gan_loss)
        self.target_model_optimizer = torch.optim.AdamW(self.target_model.parameters(), weight_decay=0.05)

        self.gan = None
        self.set_gan()
        self.gan_loss_function = Losses.get_combined_gan_loss
        self.gan_optimizer = torch.optim.AdamW(self.gan.parameters(), weight_decay=0.05)
        self.gan_scheduler = None
        self.set_gan_scheduler()
        self.gan_copy = copy.deepcopy(self.gan)

        self.recovered_from_nan_target_model = 0
        self.recovered_from_nan_gan = 0
        self.gradient_clipping_coefficient = 1.0

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
                    nr_classes=self.nr_classes,
                    device=self.device,
                    nr_channels=self.training_hyperparameters.nr_channels,
                    is_batch_norm=False,
                    is_weight_norm=True,
                    dim=f"{self.nr_dimensions}d",
                    residual_units_number=self.training_hyperparameters.target_model_residual_units_number
                )
            else:
                self.target_model = network(
                    nr_classes=self.nr_classes,
                    nr_channels=self.training_hyperparameters.nr_channels,
                    is_batch_norm=False,
                    is_weight_norm=True,
                    dim=f"{self.nr_dimensions}d",
                )
            break

    def set_gan(self):
        self.gan = Gan(
            nr_channels=self.training_hyperparameters.nr_channels,
            is_batch_norm=True,
            device=self.device,
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

    def load_models_if_present(self, path_to_load_target_model, path_to_load_gan):
        for model, path_to_load in [
            (self.target_model, path_to_load_target_model),
            (self.gan, path_to_load_gan),
        ]:
            model = model.to(self.device)

            if path_to_load is not None:
                ld = torch.load(path_to_load, map_location=self.device)
                model.load_state_dict(ld)
