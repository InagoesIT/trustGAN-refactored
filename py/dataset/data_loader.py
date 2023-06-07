import torch
import os

from py.dataset.modifier import Modifier


class DataLoader:
    @staticmethod
    def get_processed_input_and_labels(path_to_dataset, nr_classes, dataset_type):
        x = torch.load(os.path.join(path_to_dataset, f"x_{dataset_type}.pt"))
        y = torch.load(os.path.join(path_to_dataset, f"y_{dataset_type}.pt"))
        x = x.to(torch.float)
        y = y.to(torch.long)

        actual_classes_nr = (y >= nr_classes).sum()
        if actual_classes_nr > 0:
            print("WARNING: more classes than asked for")
            y[actual_classes_nr] = 0

        y = torch.nn.functional.one_hot(y, num_classes=nr_classes)
        y = torch.cat([y[..., i][:, None, ...] for i in range(y.shape[-1])], dim=1)
        y_minus_one_plus_one = Modifier.convert_from_one_hot_to_minus_one_plus_one_encoding(y)

        return x, y, y_minus_one_plus_one

    @staticmethod
    def get_dataloader(dataset, batch_size=64, use_cuda=True):
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs,
        )

        return loader
