from dataset.data_loader import DataLoader

from dataset.dataset_handler import DatasetHandler


class DataLoaders:
    def __init__(self, path_to_dataset, nr_classes, batch_size, k_fold):
        self.path_to_dataset = path_to_dataset
        self.batch_size = batch_size
        self.k_fold = k_fold
        self.nr_classes = nr_classes
        self.validation = []
        self.train = []
        self.test = None
        self.set_data_loaders()

    def set_data_loaders(self):
        self.set_validation_and_training_data_loaders()
        self.set_test_data_loader()
        
    def set_validation_and_training_data_loaders(self):
        input_items, labels = DataLoader.get_processed_input_and_labels(
                path_to_dataset=self.path_to_dataset,
                nr_classes=self.nr_classes,
                dataset_type="trainvalidation")

        dataset_handler = DatasetHandler(input_items, labels)
        total_size = len(dataset_handler)
        fold_size = int(total_size / self.k_fold)
        
        if self.k_fold < 2:
            print("INFO: The test and validation set will be split 80/20, there won't be any cross-validation.")
            fold_size = int(total_size * 0.2)

        for fold_index in range(self.k_fold):
            train_set, validation_set = DataLoaders.get_train_and_validation_dataset(
                dataset_handler, fold_index, fold_size, total_size)

            self.train.append(
                DataLoader.get_dataloader(
                    dataset=train_set,
                    batch_size=self.batch_size))
            self.validation.append(
                DataLoader.get_dataloader(
                    dataset=validation_set,
                    batch_size=self.batch_size))

    @staticmethod
    def get_train_and_validation_dataset(dataset_handler, fold_index, fold_size, total_size):
        validation_start_index = fold_index * fold_size
        validation_end_index = validation_start_index + fold_size
        validation_indices = list(range(validation_start_index, validation_end_index))

        train_left_indices = list(range(0, validation_start_index))
        train_right_indices = list(range(validation_end_index, total_size))
        train_indices = train_left_indices + train_right_indices

        train_data = dataset_handler.data[train_indices]
        train_label = dataset_handler.label[train_indices]
        train_set = DatasetHandler(train_data, train_label)

        validation_data = dataset_handler.data[validation_indices]
        validation_label = dataset_handler.label[validation_indices]
        validation_set = DatasetHandler(validation_data, validation_label)

        return train_set, validation_set

    def set_test_data_loader(self):
        input_items, labels = DataLoader.get_processed_input_and_labels(
            path_to_dataset=self.path_to_dataset,
            nr_classes=self.nr_classes,
            dataset_type="test")
        dataset = DatasetHandler(input_items, labels)

        self.test = DataLoader.get_dataloader(
            dataset=dataset,
            batch_size=self.batch_size)
