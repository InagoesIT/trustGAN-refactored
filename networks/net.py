import torch
import numpy as np

from networks.residual_network_unit import ResidualNetworkUnit


class Net(torch.nn.Module):
    def __init__(
        self,
        nr_classes,
        nr_channels,
        device,
        kernel_size=3,
        residual_units_number=7,
        is_weight_norm=True,
        is_batch_norm=False,
        dilation_coef=2,
        dim="2d",
    ):
        super(Net, self).__init__()
        self.device = device

        #
        self.nr_dimensions = int(dim[:-1])
        print(f"INFO: nr_dims = {self.nr_dimensions}")
        if dim == "1d":
            convolution = torch.nn.Conv1d
        elif dim == "2d":
            convolution = torch.nn.Conv2d

        #
        residual_units = [pow(base=2, exp=exponent) for exponent in range(residual_units_number)]
        channels = np.array(residual_units)
        channels = channels[channels > nr_channels]
        channels = np.append([nr_channels], channels)
        print(f"INFO: Using these different channel steps {channels}")

        self.layers = torch.nn.ModuleList(
            [
                ResidualNetworkUnit(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    device=self.device,
                    kernel_size=kernel_size,
                    dilation=dilation_coef**i,
                    is_weight_norm=is_weight_norm,
                    is_batch_norm=is_batch_norm,
                    dim=dim,
                )
                for i in range(len(channels) - 1)
            ]
        )

        # This convolution is very important. Without it the target_model does not learn
        self.convolution = convolution(in_channels=channels[-1], out_channels=channels[-1], kernel_size=1)

        self.linear_layer0 = torch.nn.Linear(in_features=channels[-1], out_features=residual_units[-1])
        self.linear_layer1 = torch.nn.Linear(in_features=residual_units[-1], out_features=nr_classes)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        for unit in self.layers:
            x = unit(x)
        x = self.convolution(x)

        for _ in range(self.nr_dimensions):
            x, _ = torch.max(x, dim=-1)

        x = self.dropout(self.relu(self.linear_layer0(x)))
        x = self.linear_layer1(x)

        return x

