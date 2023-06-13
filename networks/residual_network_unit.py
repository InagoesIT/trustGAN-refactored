import torch


class ResidualNetworkUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        device,
        dilation=1,
        dim="2d",
        is_weight_norm=True,
        is_batch_norm=False,
        is_gan=False,
    ):
        super(ResidualNetworkUnit, self).__init__()
        self.device = device

        if dim == "1d":
            conv = torch.nn.Conv1d
            batchnorm = torch.nn.BatchNorm1d
        elif dim == "2d":
            conv = torch.nn.Conv2d
            batchnorm = torch.nn.BatchNorm2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.convolution1 = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )
        torch.nn.init.constant_(self.convolution1.bias, 0.0)

        self.convolution2 = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode="replicate",
        )  # padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.convolution2.bias, 0.0)

        self.convolution3 = conv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            padding_mode="replicate",
        )  # padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.convolution3.bias, 0.0)

        # Helps a lot the GAN
        if is_gan:
            self.relu = torch.nn.LeakyReLU()
        else:
            self.relu = torch.nn.ReLU()

        if is_weight_norm:
            self.convolution2 = torch.nn.utils.weight_norm(self.convolution2)
            self.convolution3 = torch.nn.utils.weight_norm(self.convolution3)

        if is_batch_norm:
            self.batch_norm = batchnorm(num_features=out_channels)
            self.batch_norm = self.batch_norm.to(self.device)
        else:
            self.batch_norm = None
        
        self.convolution1 = self.convolution1.to(self.device)
        self.convolution2 = self.convolution2.to(self.device)
        self.convolution3 = self.convolution3.to(self.device)

        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        result = self.convolution2(x)
        result = self.relu(result)

        result = self.convolution3(result)
        result = self.scale.to(self.device) * result

        x = self.convolution1(x)
        x = x + result

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.relu(x)

        return x

