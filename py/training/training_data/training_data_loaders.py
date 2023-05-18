from py.dataset.data_loader import DataLoader

from py.dataset.dataset_handler import DatasetHandler


class TrainingDataLoaders:
    def __init__(self, path_to_dataset, training_parameters):
        self.path_to_dataset = path_to_dataset
        self.training_parameters = training_parameters
        self.validation = []
        self.train = []
        self.test = None

    def set_data_loaders(self):
        self.set_validation_and_training_data_loaders()
        self.set_test_data_loader()

    def set_validation_and_training_data_loaders(self):
        input_items, labels = DataLoader.get_processed_input_and_labels(
                path2dataset=self.path_to_dataset,
                nr_classes=self.training_parameters.nr_classes,
                dataset_type="trainvalidation")

        dataset_handler = DatasetHandler(input_items, labels)
        total_size = len(dataset_handler)
        fold_size = int(total_size / self.training_parameters.k_fold)

        for fold_index in range(self.training_parameters.k_fold):
            train_set, validation_set = TrainingDataLoaders.get_train_and_validation_dataset(
                dataset_handler, fold_index, fold_size, total_size)

            self.train.append(
                DataLoader.get_dataloader(
                    dataset=train_set,
                    batch_size=self.training_parameters.batch_size))
            self.validation.append(
                DataLoader.get_dataloader(
                    dataset=validation_set,
                    batch_size=self.training_parameters.batch_size))

    @staticmethod
    def get_train_and_validation_dataset(dataset_handler: DatasetHandler, fold_index, fold_size, total_size):
        validation_start_index = fold_index * fold_size
        validation_end_index = validation_start_index + fold_size
        validation_indices = list(range(validation_start_index, validation_end_index))

        train_left_indices = list(range(0, validation_start_index))
        train_right_indices = list(range(validation_end_index, total_size))
        train_indices = train_left_indices + train_right_indices

        train_data = dataset_handler.state[train_indices]
        train_label = dataset_handler.label[train_indices]
        train_set = DatasetHandler(train_data, train_label)

        validation_data = dataset_handler.state[validation_indices]
        validation_label = dataset_handler.label[validation_indices]
        validation_set = DatasetHandler(validation_data, validation_label)

        return train_set, validation_set

    def set_test_data_loader(self):
        input_items, labels = DataLoader.get_processed_input_and_labels(
            path2dataset=self.path_to_dataset,
            nr_classes=self.training_parameters.nr_classes,
            dataset_type="test")
        dataset = DatasetHandler(input_items, labels)

        self.test = DataLoader.get_dataloader(
            dataset=dataset,
            batch_size=self.training_parameters.batch_size)
