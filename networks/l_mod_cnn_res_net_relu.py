import torch


class LModCNNResNetRelu(torch.nn.Module):
    def __init__(self, nr_classes, kernel_size=7, **kwargs):
        super(LModCNNResNetRelu, self).__init__()

        self.nr_classes = nr_classes

        nbc = 8
        self.convolution00 = torch.nn.Conv1d(in_channels=2, out_channels=nbc, kernel_size=1)
        self.convolution01 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.convolution02 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 16
        self.convolution10 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.convolution11 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.convolution12 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 32
        self.convolution20 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.convolution21 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.convolution22 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 64
        self.convolution30 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.convolution31 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.convolution32 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.linear_layer1 = torch.nn.Linear(in_features=nbc, out_features=256)
        self.linear_layer2 = torch.nn.Linear(in_features=256, out_features=self.nr_classes)

        self.dropout = torch.nn.Dropout()

        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x = self.relu(self.convolution00(x))
        y = x.clone()
        x = self.relu(self.convolution01(x))
        x = self.convolution02(x)
        x = self.relu(x + y)

        x = self.relu(self.convolution10(x))
        y = x.clone()
        x = self.relu(self.convolution11(x))
        x = self.convolution12(x)
        x = self.relu(x + y)

        x = self.relu(self.convolution20(x))
        y = x.clone()
        x = self.relu(self.convolution21(x))
        x = self.convolution22(x)
        x = self.relu(x + y)

        x = self.relu(self.convolution30(x))
        y = x.clone()
        x = self.relu(self.convolution31(x))
        x = self.convolution32(x)
        x = self.relu(x + y)

        x = x.mean(axis=-1)

        x = self.relu(self.linear_layer1(x))
        x = self.dropout(x)

        x = self.linear_layer2(x)

        return x
