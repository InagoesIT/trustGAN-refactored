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
        self.dilation = dilation
        convolution = torch.nn.Conv1d
        batch_norm = torch.nn.BatchNorm1d
        if dim == "2d":
            convolution = torch.nn.Conv2d
            batch_norm = torch.nn.BatchNorm2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.convolution1 = None
        self.init_convolution_layers(convolution)

        # Helps a lot the GAN
        if is_gan:
            self.relu = torch.nn.LeakyReLU()
        else:
            self.relu = torch.nn.ReLU()

        if is_weight_norm:
            self.convolution2 = torch.nn.utils.weight_norm(self.convolution2)
            self.convolution3 = torch.nn.utils.weight_norm(self.convolution3)

        if is_batch_norm:
            self.is_batch_norm = batch_norm(num_features=out_channels)
        else:
            self.is_batch_norm = None

        self.scale = torch.nn.Parameter(torch.zeros(1))

    def init_convolution_layers(self, convolution):
        self.convolution1 = convolution(
            in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1
        )
        torch.nn.init.constant_(self.convolution1.bias, 0.0)

        self.convolution2 = convolution(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.dilation * (self.kernel_size - 1) // 2,
            dilation=self.dilation,
            padding_mode="replicate",
        )  # padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.convolution2.bias, 0.0)

        self.convolution3 = convolution(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.dilation * (self.kernel_size - 1) // 2,
            dilation=self.dilation,
            padding_mode="replicate",
        )  # padding_mode='replicate' seem very important for GAN
        torch.nn.init.constant_(self.convolution3.bias, 0.0)

    def forward(self, input_tensor):
        residual = self.convolution2(input_tensor)
        residual = self.relu(residual)

        residual = self.convolution3(residual)
        # balancing the importance of the input features
        # with the learned values in the residual connection
        residual = self.scale * residual

        output = self.convolution1(input_tensor)
        output = output + residual

        if self.is_batch_norm is not None:
            output = self.is_batch_norm(output)

        output = self.relu(output)

        return output

