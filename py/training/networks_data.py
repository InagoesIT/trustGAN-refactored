import pkgutil

import torch
import copy

from py.training.networks.gan import Gan
from py.training.losses import get_softmax_cross_entropy_loss, get_combined_gan_loss

package = __import__("py.training.networks")


class NetworksData:
    def __init__(self, path2net, nr_dims, training_params):
        if path2net is not None:
            self.net = path2net
        else:
            # find all the modules in the package
            for importer, modname, is_pkg in pkgutil.walk_packages(package.__path__):
                # import the module
                module = importer.find_module(modname).load_module(modname)

                # check if the module has the class
                if hasattr(module, training_params.network_name):
                    # get the class from the module
                    network = getattr(module, training_params.network_name)
                    self.net = network(
                        nr_classes=training_params.nr_classes,
                        nr_channels=training_params.nr_channels,
                        is_batch_norm=False,
                        is_weight_norm=True,
                        dim=f"{nr_dims}d",
                    )
                    break
        self.net_loss = get_softmax_cross_entropy_loss
        self.net_optim = torch.optim.AdamW(self.net.parameters(), weight_decay=0.05)

        self.gan = Gan(
            nr_channels=training_params.nr_channels,
            is_batch_norm=True,
            is_weight_norm=False,
            dim=f"{nr_dims}d",
        )
        self.gan_loss = get_combined_gan_loss
        self.gan_optim = torch.optim.AdamW(self.gan.parameters(), weight_decay=0.05)
        self.gan_scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.gan_optim,
            base_lr=1.0e-3,
            max_lr=5.0e-3,
            step_size_up=50,
            cycle_momentum=False,
        )

        self.recovered_from_nan_net = 0
        self.recovered_from_nan_gan = 0
        self.grad_clipping_coeff = 1.0
        self.gan2 = copy.deepcopy(self.gan)
