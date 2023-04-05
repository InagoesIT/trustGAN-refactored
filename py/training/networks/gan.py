import torch
import numpy as np

from py.training.networks.res_net_unit import ResNetUnit


class Gan(torch.nn.Module):
    def __init__(
            self,
            nr_channels,
            kernel_size=3,
            residual_units=None,
            is_weight_norm=True,
            is_batch_norm=False,
            dilation_coefficient=2,
            dim="2d",
    ):
        super(Gan, self).__init__()

        if residual_units is None:
            residual_units = [1, 2, 4, 8, 16]

        convolution = torch.nn.Conv1d
        if dim == "2d":
            convolution = torch.nn.Conv2d

        channel_steps = np.array(residual_units)
        channel_steps = channel_steps[channel_steps > nr_channels]
        channel_steps = np.append([nr_channels], channel_steps)
        print(f"INFO: Using these different channel steps {channel_steps}")

        self.layers = torch.nn.ModuleList(
            [
                ResNetUnit(
                    in_channels=channel_steps[i],
                    out_channels=channel_steps[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation_coefficient ** i,
                    is_weight_norm=is_weight_norm,
                    is_batch_norm=is_batch_norm,
                    is_gan=True,
                    dim=dim,
                )
                for i in range(len(channel_steps) - 1)
            ]
        )

        self.convolution1 = convolution(in_channels=channel_steps[-1], out_channels=channel_steps[0], kernel_size=1)

    def forward(self, x):
        for unit in self.layers:
            x = unit(x)

        x = self.convolution1(x)
        x = torch.tanh(x)

        return x
