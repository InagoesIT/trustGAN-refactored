# Authors:
#   Helion du Mas des Bourboux <helion.dumasdesbourboux'at'thalesgroup.com>
#
# MIT License
#
# Copyright (c) 2022 THALES
#   All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# 2022 october 21

import torch
import numpy as np


class WaveUnet(torch.nn.Module):
    def __init__(
        self,
        nr_channels,
        nr_classes,
        residual_units=None,
        kernel_size_down=3,
        kernel_size_up=3,
        dim="2d",
        scale_factor=2,
        **kwargs,
    ):
        super(WaveUnet, self).__init__()

        if residual_units is None:
            residual_units = [
                8,
                9,
                16,
                17,
                32,
                33,
                64,
                64,
                128,
                129,
            ]
        self.nr_dimensions = int(dim[:-1])
        print(f"INFO: nr_dims = {self.nr_dimensions}")
        if dim == "1d":
            conv = torch.nn.Conv1d
            batchnorm = torch.nn.BatchNorm1d
            maxpool = torch.nn.MaxPool1d
        elif dim == "2d":
            conv = torch.nn.Conv2d
            batchnorm = torch.nn.BatchNorm2d
            maxpool = torch.nn.MaxPool2d

        #
        chs = np.array(residual_units)
        chs = chs[chs >= nr_channels]

        self.convolutionIn = conv(in_channels=nr_channels, out_channels=chs[0], kernel_size=1)

        self.downLayers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    conv(
                        in_channels=chs[i],
                        out_channels=chs[i + 1],
                        kernel_size=kernel_size_down,
                        padding=(kernel_size_down - 1) // 2,
                    ),
                    batchnorm(chs[i + 1]),
                    torch.nn.ReLU(),
                )
                for i in range(chs.size - 1)
            ]
        )

        self.ubottom = torch.nn.Identity()
        self.upLayers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    conv(
                        in_channels=2 * chs[i],
                        out_channels=chs[i - 1],
                        kernel_size=kernel_size_up,
                        padding=(kernel_size_up - 1) // 2,
                    ),
                    batchnorm(chs[i - 1]),
                    torch.nn.ReLU(),
                )
                for i in range(1, chs.size)[::-1]
            ]
        )

        nr_chs_end = 2 * chs[0] + nr_channels
        self.convOut_0 = conv(
            in_channels=nr_chs_end, out_channels=nr_chs_end, kernel_size=1
        )
        self.convOut_1 = conv(
            in_channels=nr_chs_end, out_channels=nr_chs_end, kernel_size=1
        )
        self.convOut_2 = conv(
            in_channels=nr_chs_end, out_channels=nr_classes, kernel_size=1
        )

        self.relu = torch.nn.ReLU()
        self.maxpool = maxpool(kernel_size=scale_factor, ceil_mode=True)
        self.upsample = torch.nn.Upsample(scale_factor=scale_factor)

        print(f"INFO: Minimum meaningful size = {self.get_minimum_size(nr_channels)}")

    def forward(self, x_in):
        keep_x = x_in.clone()
        x = self.convolutionIn(x_in)
        trans_x = self.relu(x)

        x, saved_layers, sizes = self.forward_down(x)
        x = self.ubottom(x)
        x = self.forward_up(x, saved_layers, sizes)

        x = torch.cat([x, keep_x, trans_x], axis=1)
        x = self.relu(self.convOut_0(x))
        x = self.relu(self.convOut_1(x))
        x = self.convOut_2(x)

        return x

    def forward_down(self, x):
        sizes = [x.shape[-1]]
        saved_layers = []

        for i, layer in enumerate(self.downLayers):
            x = layer(x)

            saved_layers += [x.clone()]
            if i < len(self.downLayers) - 1:
                x = self.maxpool(x)
                sizes += [x.shape[-1]]

        return x, saved_layers, sizes

    def forward_up(self, x, saved_layers, sizes):
        for i, layer in enumerate(self.upLayers):
            idx_up = len(self.upLayers) - 1 - i
            idx_do = len(self.downLayers) - 1 - i

            if sizes[idx_up] != x.shape[-1]:
                x = self.upsample(x)
            if x.shape[-1] > saved_layers[idx_do].shape[-1]:
                if x.ndim == 4:
                    x = x[
                        ...,
                        : saved_layers[idx_do].shape[-1],
                        : saved_layers[idx_do].shape[-1],
                    ]
                elif x.ndim == 3:
                    x = x[..., : saved_layers[idx_do].shape[-1]]

            x = torch.cat([x, saved_layers[idx_do]], axis=1)
            x = layer(x)

        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_minimum_size(self, nr_channels):
        self.eval()
        length = 0
        bottom_length = 1
        while bottom_length == 1:
            length += 1
            if self.nr_dimensions == 1:
                x = torch.rand((1, nr_channels, length), device=self.device)
            elif self.nr_dimensions == 2:
                x = torch.rand((1, nr_channels, length, length), device=self.device)
            x = self.convolutionIn(x)
            _, _, sizes = self.forward_down(x)
            bottom_length = sizes[-1]

        self.train()

        return length
