import torch
import numpy as np

from networks.residual_network_unit import ResidualNetworkUnit


class Gan(torch.nn.Module):
    def __init__(
        self,
        nr_channels,
        device,
        kernel_size=3,
        residual_units_number=5,
        is_weight_norm=True,
        is_batch_norm=False,
        dilation_coefficient=2,
        dim="2d",
    ):
        super(Gan, self).__init__()
        self.device = device

        if dim == "1d":
            conv = torch.nn.Conv1d
        elif dim == "2d":
            conv = torch.nn.Conv2d

        residual_units = [pow(base=2, exp=exponent) for exponent in range(residual_units_number)]
        chs = np.array(residual_units)
        chs = chs[chs > nr_channels]
        chs = np.append([nr_channels], chs)
        print(f"INFO: Using these different channel steps {chs}")

        self.layers = torch.nn.ModuleList(
            [
                ResidualNetworkUnit(
                    in_channels=chs[i],
                    out_channels=chs[i + 1],
                    kernel_size=kernel_size,
                    device=self.device,
                    dilation=dilation_coefficient ** i,
                    is_weight_norm=is_weight_norm,
                    is_batch_norm=is_batch_norm,
                    is_gan=True,
                    dim=dim,
                )
                for i in range(len(chs) - 1)
            ]
        )        
        self.conv1 = conv(in_channels=chs[-1], out_channels=chs[0], kernel_size=1)
        self.conv1 = self.conv1.to(torch.device("cuda:0"))
        self.to(device)

    def forward(self, x):
        x = x.to(torch.device("cuda:0"))
        for unit in self.layers:
            x = unit(x)

        x = self.conv1(x)
        x = torch.tanh(x)

        return x
