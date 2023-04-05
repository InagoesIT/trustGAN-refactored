from py.dataset.data_loader import DataLoader


class DataLoaders:
    def __init__(self, path2dataset, training_params, verbose):
        self.train = DataLoader.get_dataloader(
            path2dataset=path2dataset,
            nr_classes=training_params.nr_classes,
            dataset_type="train",
            batch_size=training_params.batch_size,
            verbose=verbose,
        )
        self.valid = DataLoader.get_dataloader(
            path2dataset=path2dataset,
            nr_classes=training_params.nr_classes,
            dataset_type="valid",
            batch_size=training_params.batch_size,
            verbose=verbose,
        )
        self.test = DataLoader.get_dataloader(
            path2dataset=path2dataset,
            nr_classes=training_params.nr_classes,
            dataset_type="test",
            batch_size=training_params.batch_size,
            verbose=verbose,
        )
