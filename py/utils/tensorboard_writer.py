import glob
import os
import numpy as np
import re
from torch.utils.tensorboard import SummaryWriter

from py.performances.performances_logger import PerformancesLogger


class TensorboardWriter:
    def __init__(self, path_to_root_folder, total_epochs, validation_interval, path_to_performances):
        self.base_name_for_model_performances = PerformancesLogger.base_name_for_model_performances_file
        self.path_to_root_folder = path_to_root_folder
        self.total_epochs = total_epochs
        self.validation_interval = validation_interval
        self.writer = SummaryWriter(f"{self.path_to_root_folder}/tensorboard")

        self.average_performances = np.load("{}/{}".format(self.path_to_root_folder, path_to_performances),
                                            allow_pickle=True)
        self.average_performances = self.average_performances.item()
        self.train_metrics = list(self.average_performances[list(self.average_performances.keys())[0]].keys())

    def write_performances_to_tensorboard(self, performance_label, performances, start_index=0):
        train_metrics = list(performances[list(performances.keys())[0]].keys())
        performances_size = 0

        for metric in train_metrics:
            for dataset_type in performances.keys():
                performances_for_metric = performances[dataset_type][metric]
                epochs = [epoch for epoch in range(self.total_epochs)]
                if len(performances_for_metric) != self.total_epochs + 1:
                    epochs = PerformancesLogger.get_validation_epochs(epochs=epochs,
                                                                      validation_at=self.validation_interval,
                                                                      total_epochs=self.total_epochs)
                else:
                    performances_size = len(performances_for_metric)
                for index in range(len(epochs)):
                    if performance_label.find("model") == -1:
                        self.writer.add_scalar(f'{performance_label};{dataset_type};{metric}',
                                          performances_for_metric[index + start_index], epochs[index])
                    else:
                        self.writer.add_scalars("models", {
                            f'{performance_label};{dataset_type};{metric}': performances_for_metric[index]},
                                           epochs[index])
        return performances_size        

    def plot_execution_time(self):
        time_data_path = "{}/execution_data.npy".format(self.path_to_root_folder)
        if not os.path.exists(time_data_path):
            return
        time_data = np.load(time_data_path, allow_pickle=True)
        time_data = time_data.item()
        epochs = [epoch for epoch in range(self.total_epochs)]
        self.writer.add_scalar(f'execution_data', epochs, time_data["time"])

    def plot_models(self):
        for filename in glob.glob(os.path.join(self.path_to_root_folder, f'*{self.base_name_for_model_performances}*')):
            model_index = re.findall(r'\d+', filename)[-1]
            performance_label = f"model_{model_index}"
            performances = np.load(filename,
                                   allow_pickle=True)
            performances = performances.item()
            performances_size = self.write_performances_to_tensorboard(performance_label=performance_label,
                                                   performances=performances)
        return performances_size

    def plot_model_performances(self, performance_label, start_index=0):
        self.write_performances_to_tensorboard(performance_label=performance_label,
                                               performances=self.average_performances,
                                               start_index=start_index)
                                            

    def plot_models_together(self):
        performances_size = self.plot_models()
        start_index = 0
        if performances_size != len(self.average_performances["training"]["loss_gan"]):
            start_index = performances_size
        self.plot_model_performances(performance_label="average_model", start_index=start_index)

    def plot_models_separately(self):
        self.plot_models()
        self.plot_model_performances(performance_label="average")
