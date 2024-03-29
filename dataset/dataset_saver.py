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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, ExPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# 2022 october 21

import os

import numpy
import numpy as np
import tempfile
import torch
import torchvision
import torchvision.transforms


class DatasetSaver:
    def __init__(self, dataset, path_to_root_folder, seed=42):
        self.dataset = dataset
        self.path_to_root_folder = path_to_root_folder
        DatasetSaver.set_seeds(seed)
        data = self.get_data()
        self.save_data(data)

    @staticmethod
    def set_seeds(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_data(self):
        if hasattr(torchvision.datasets, self.dataset):
            data = self.get_torch_dataset()
        else:
            print(f"ERROR: did not find the dataset {self.dataset}")
            raise AttributeError

        for k, v in data.items():
            nr_unique_elements = torch.unique(v["y"]).size()
            print(
                f"INFO: {k} has x shape = {v['x'].shape} and y shape = {v['y'].shape}, nr uniques = {nr_unique_elements}"
            )

        return data

    def get_torch_dataset(self):
        """ Returns data with trainvalid and test keys"""
        data = self.get_train_and_validation()

        # Test
        test_set = self.get_torch_dataset_with_type(is_train=False)
        data["test"] = test_set

        # Calculate the dataset train-test split
        splits = np.array(
            [data[el]["y"].shape[0] for el in ["trainvalidation", "test"]], dtype=float
        )
        splits /= splits.sum()
        print("INFO: splits:", splits)

        return data

    def get_train_and_validation(self):
        data = {}
        train_valid_set = self.get_torch_dataset_with_type(is_train=True)

        # shuffle elements
        nr_samples = train_valid_set["y"].shape[0]
        random_indexes = np.arange(nr_samples)
        np.random.shuffle(random_indexes)

        data["trainvalidation"] = {
            "x": train_valid_set["x"][random_indexes],
            "y": train_valid_set["y"][random_indexes],
        }

        return data

    def get_torch_dataset_with_type(self, is_train: bool):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            if self.dataset in ["OxfordIIITPet"]:
                kwargs = {"target_types": "segmentation"}
            else:
                kwargs = {}
            data = (
                getattr(torchvision.datasets, self.dataset)(
                    tmp_dir_name,
                    is_train,
                    download=True,
                    transform=torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                        ]
                    ),
                    **kwargs,
                ),
            )

            x, y = self.get_x_y_from_data(data)
            x, y = self.channel_manipulations(x, y)

        return {"x": x, "y": y}

    def get_x_y_from_data(self, data):
        # doesn't need processing
        if hasattr(data[0], "data") and hasattr(data[0], "targets"):
            x, y = data[0].data, data[0].targets
        else:
            x, y = self.get_data_from_loaders(data[0])

        # make x, y to be tensors
        if "numpy" in str(type(x)):
            x = torch.from_numpy(x)
        if "numpy" in str(type(y)):
            y = torch.from_numpy(y)
        elif type(y) is list:
            y = torch.from_numpy(np.array(y))

        return x, y

    @staticmethod
    def get_data_from_loaders(data):
        x = []
        y = []
        dataset_size = 10

        for i, tmp_data in enumerate(data):
            if i >= dataset_size:
                break

            # add a new dimension to tmp_x
            tmp_x = tmp_data[0][None, ...]
            tmp_y = tmp_data[1]

            # we have an image
            if "PIL.PngImagePlugin.PngImageFile" in str(type(tmp_y)):
                tmp_y = torch.from_numpy(np.asarray(tmp_y).copy())[None, ...]

            x += [tmp_x]
            y += [tmp_y]

        x = DatasetSaver.resize_x(x)
        y = DatasetSaver.resize_y(y)

        return x, y

    @staticmethod
    def resize_x(x):
        if x[0].ndim == 4:
            x = [
                torchvision.transforms.Resize(
                    size=(64, 64),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                )(el) for el in x
            ]
        # convert from list of tensors to tensors
        return torch.cat(x)

    @staticmethod
    def resize_y(y):
        if y[0].ndim == 3:
            y = [
                torchvision.transforms.Resize(
                    size=(64, 64),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                )(el) for el in y
            ]
        y = torch.cat(y)

        # encode the values in y to be from 0 to nr(unique_values)
        for i, val in enumerate(torch.sort(torch.unique(y))[0]):
            y_eq_to_val = y == val
            y[y_eq_to_val] = i

        return y

    def channel_manipulations(self, x, y):
        # add a dimension m (n, m, n)
        if self.dataset in ["MNIST", "FashionMNIST"]:
            x = torch.unsqueeze(x, 1)
        # transposing a tensor x with shape (batch_size, height, width, channels)
        # to (batch_size, channels, height, width)
        elif self.dataset in ["CIFAR10"]:
            x = torch.cat([x[..., i][:, None, ...] for i in range(x.shape[-1])], dim=1)

        return x, y

    def save_data(self, data):
        for k, v in data.items():
            os.makedirs(os.path.join(self.path_to_root_folder, self.dataset), exist_ok=True)
            for n_data, a_data in v.items():
                tmp_path2save = os.path.join(
                    self.path_to_root_folder, self.dataset, f"{n_data}_{k}.pt"
                )
                torch.save(a_data, tmp_path2save)
