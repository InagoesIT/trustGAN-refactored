import torch
import os

from py.dataset.dataset_handler import DatasetHandler


class DataLoader:
    @staticmethod
    def get_dataloader(path2dataset, nr_classes, dataset_type, batch_size=64, verbose=True):
        use_cuda = torch.cuda.is_available()
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        x = torch.load(os.path.join(path2dataset, f"x_{dataset_type}.pt"))
        y = torch.load(os.path.join(path2dataset, f"y_{dataset_type}.pt"))
        x = x.to(torch.float)
        y = y.to(torch.long)

        actual_classes_nr = (y >= nr_classes).sum()
        if actual_classes_nr > 0:
            print("WARNING: more classes than asked for")
            y[actual_classes_nr] = 0

        y = torch.nn.functional.one_hot(y, num_classes=nr_classes)
        y = torch.cat([y[..., i][:, None, ...] for i in range(y.shape[-1])], dim=1)

        loader = torch.utils.data.DataLoader(
            DatasetHandler(x, y),
            batch_size=batch_size,
            shuffle=True,
            **kwargs,
        )

        return loader
