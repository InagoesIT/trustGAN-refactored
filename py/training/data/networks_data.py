import pkgutil

import torch
import copy

from py.networks.gan import Gan
from py.performances.losses import get_softmax_cross_entropy_loss, get_combined_gan_loss

package = __import__("py.networks")


class NetworksData:
    def __init__(self, nr_dims, training_hyperparameters, given_target_model=None):
        if given_target_model is not None:
            self.target_model = given_target_model
        else:
            for importer, modname, is_pkg in pkgutil.walk_packages(package.__path__):
                module = importer.find_module(modname).load_module(modname)

                if hasattr(module, training_hyperparameters.target_model_network_type):
                    network = getattr(module, training_hyperparameters.target_model_network_type)
                    self.target_model = network(
                        nr_classes=training_hyperparameters.nr_classes,
                        nr_channels=training_hyperparameters.nr_channels,
                        is_batch_norm=False,
                        is_weight_norm=True,
                        dim=f"{nr_dims}d",
                    )
                    break
        self.target_model_loss_type = get_softmax_cross_entropy_loss
        self.target_model_optimizer = torch.optim.AdamW(self.target_model.parameters(), weight_decay=0.05)

        self.gan = Gan(
            nr_channels=training_hyperparameters.nr_channels,
            is_batch_norm=True,
            is_weight_norm=False,
            dim=f"{nr_dims}d",
        )
        self.gan_loss_type = get_combined_gan_loss
        self.gan_optimizer = torch.optim.AdamW(self.gan.parameters(), weight_decay=0.05)
        self.gan_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.gan_optimizer,
            base_lr=1.0e-3,
            max_lr=5.0e-3,
            step_size_up=50,
            cycle_momentum=False,
        )

        self.recovered_from_nan_target_model = 0
        self.recovered_from_nan_gan = 0
        self.grad_clipping_coeff = 1.0
        self.gan_copy = copy.deepcopy(self.gan)
