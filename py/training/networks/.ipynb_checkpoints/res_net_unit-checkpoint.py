import torch


class ResNetUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        dim="2d",
        is_weight_norm=True,
        is_batch_norm=False,
        is_gan=False,
    ):
        super(ResNetUnit, self).__init__()

        if dim == "1d":
            conv = torch.nn.Conv1d
            batchnorm = torch.nn.BatchNorm1d
        elif dim == "2d":
            conv = torch.nn.Conv2d
            batchnorm = torch.nn.BatchNorm2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        torch.nn.init.constant_(self.conv1.bias, 0.0)

        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode="replicate",
        )  # padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.conv2.bias, 0.0)

        self.conv3 = conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode="replicate",
        )  # padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.conv3.bias, 0.0)

        # Helps a lot the GAN
        if is_gan:
            self.relu = torch.nn.LeakyReLU()
        else:
            self.relu = torch.nn.ReLU()

        if is_weight_norm:
            self.conv2 = torch.nn.utils.weight_norm(self.conv2)
            self.conv3 = torch.nn.utils.weight_norm(self.conv3)

        if is_batch_norm:
            self.batch_norm = batchnorm(num_features=out_channels)
            self.batch_norm = self.batch_norm.to(torch.device("cuda:0"))
        else:
            self.batch_norm = None
        
        self.conv1 = self.conv1.to(torch.device("cuda:0"))
        self.conv2 = self.conv2.to(torch.device("cuda:0"))
        self.conv3 = self.conv3.to(torch.device("cuda:0"))

        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        res = self.conv2(x)
        res = self.relu(res)

        res = self.conv3(res)
        res = self.scale.to(torch.device("cuda:0")) * res

        x = self.conv1(x)
        x = x + res

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.relu(x)

        return x

