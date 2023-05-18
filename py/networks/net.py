import torch
import numpy as np

from py.networks.res_net_unit import ResNetUnit


class Net(torch.nn.Module):
    def __init__(
        self,
        nr_classes,
        nr_channels,
        kernel_size=3,
        fcl=64,
        residual_units=[1, 2, 4, 8, 16, 32, 64],
        is_weight_norm=True,
        is_batch_norm=False,
        dilation_coef=2,
        dim="2d",
    ):
        super(Net, self).__init__()

        #
        self.nr_dims = int(dim[:-1])
        print(f"INFO: nr_dims = {self.nr_dims}")
        if dim == "1d":
            conv = torch.nn.Conv1d
        elif dim == "2d":
            conv = torch.nn.Conv2d

        #
        chs = np.array(residual_units)
        chs = chs[chs > nr_channels]
        chs = np.append([nr_channels], chs)
        print(f"INFO: Using these different chanel steps {chs}")

        self.layers = torch.nn.ModuleList(
            [
                ResNetUnit(
                    in_channels=chs[i],
                    out_channels=chs[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation_coef**i,
                    is_weight_norm=is_weight_norm,
                    is_batch_norm=is_batch_norm,
                    dim=dim,
                )
                for i in range(len(chs) - 1)
            ]
        )

        # This convolution is very important. Without the target_model does not learn
        self.conv1 = conv(in_channels=chs[-1], out_channels=chs[-1], kernel_size=1)

        self.lin_00 = torch.nn.Linear(in_features=chs[-1], out_features=fcl)
        self.lin_01 = torch.nn.Linear(in_features=fcl, out_features=nr_classes)

        self.relu = torch.nn.ReLU()
        self.drp = torch.nn.Dropout(0.1)

    def forward(self, x):
        for unit in self.layers:
            x = unit(x)
        x = self.conv1(x)

        for _ in range(self.nr_dims):
            x, _ = torch.max(x, dim=-1)

        x = self.drp(self.relu(self.lin_00(x)))
        x = self.lin_01(x)

        return x

