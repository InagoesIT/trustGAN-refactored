import torch


class LModCNNResNetRelu(torch.nn.Module):
    def __init__(self, nr_classes, kernel_size=7, **kwargs):
        super(LModCNNResNetRelu, self).__init__()

        self.nr_classes = nr_classes

        nbc = 8
        self.conv00 = torch.nn.Conv1d(in_channels=2, out_channels=nbc, kernel_size=1)
        self.conv01 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv02 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 16
        self.conv10 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.conv11 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv12 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 32
        self.conv20 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.conv21 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv22 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        nbc = 64
        self.conv30 = torch.nn.Conv1d(
            in_channels=nbc // 2, out_channels=nbc, kernel_size=1
        )
        self.conv31 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv32 = torch.nn.Conv1d(
            in_channels=nbc,
            out_channels=nbc,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.lin_1 = torch.nn.Linear(in_features=nbc, out_features=256)
        self.lin_2 = torch.nn.Linear(in_features=256, out_features=self.nr_classes)

        self.drp1 = torch.nn.Dropout()

        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv00(x))
        y = x.clone()
        x = self.relu(self.conv01(x))
        x = self.conv02(x)
        x = self.relu(x + y)

        x = self.relu(self.conv10(x))
        y = x.clone()
        x = self.relu(self.conv11(x))
        x = self.conv12(x)
        x = self.relu(x + y)

        x = self.relu(self.conv20(x))
        y = x.clone()
        x = self.relu(self.conv21(x))
        x = self.conv22(x)
        x = self.relu(x + y)

        x = self.relu(self.conv30(x))
        y = x.clone()
        x = self.relu(self.conv31(x))
        x = self.conv32(x)
        x = self.relu(x + y)

        x = x.mean(axis=-1)

        x = self.relu(self.lin_1(x))
        x = self.drp1(x)

        x = self.lin_2(x)

        return x
