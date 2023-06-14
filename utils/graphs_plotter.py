import os

import matplotlib.pyplot as plt
import numpy as np

from performances.performances_logger import PerformancesLogger
from training.components.paths import Paths


class GraphsPlotter:
    def __init__(self, root_folder, total_epochs, validation_interval, path_to_performances):
        self.root_folder = root_folder
        self.total_epochs = total_epochs
        self.validation_interval = validation_interval
        if path_to_performances is None:
            return
        self.average_performances = np.load("{}/{}".format(self.root_folder, path_to_performances), allow_pickle=True)
        self.average_performances = self.average_performances.item()
        self.train_metrics = list(self.average_performances[list(self.average_performances.keys())[0]].keys()) 
        Paths.process_root_folder(root_folder)

    @staticmethod
    def transform_metric_variable_to_title(metric):
        metric_title = ""
        metric_title += metric[0].upper()
        for index in range(1, len(metric)):
            if metric[index] == "_":
                metric_title += " "
            elif metric_title[index - 1] == " ":
                metric_title += metric[index].upper()
            else:
                metric_title += metric[index]        
        return metric_title

    def plot_performances(self):           
        for metric in self.train_metrics:
            for dataset_type in self.average_performances.keys():
                performances_for_metric = self.average_performances[dataset_type][metric]
                epochs = [epoch for epoch in range(self.total_epochs + 1)]
                if len(performances_for_metric) != self.total_epochs + 1:
                    epochs = PerformancesLogger.get_validation_epochs(epochs=epochs,
                                                                validation_at=self.validation_interval,
                                                                total_epochs=self.total_epochs)
                plt.plot(epochs, performances_for_metric, label=dataset_type, markersize=0.7)                
            
            metric_name = GraphsPlotter.transform_metric_variable_to_title(metric)    
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid()
            plt.savefig("{}/performance-plots/{}.png".format(self.root_folder, metric))
            plt.clf()

    def plot_execution_time(self):
        time_data_path = "{}/execution_data.npy".format(self.root_folder)
        if not os.path.exists(time_data_path):
            return
        time_data = np.load(time_data_path, allow_pickle=True)
        time_data = time_data.item()
        
        epochs = [epoch for epoch in range(self.total_epochs)]
        plt.plot(epochs, time_data["time"])
        plt.xlabel("Epoch")
        plt.ylabel("Execution time in minutes")
        plt.grid()
        plt.savefig("{}/plots/time_plot.png".format(self.root_folder))
        plt.clf()

    def plot_gpu_usage(self):
        if not os.path.exists(self.root_folder):
            os.mkdir(self.root_folder)
        data = dict()
        for root, dirs, files in os.walk(self.root_folder):
            for filename in files:
                if "execution" not in filename:
                    continue
                full_path = os.path.join(root, filename)
                execution_data = np.load(full_path, allow_pickle=True)
                gpu_usage = execution_data.item()["memory"][0] / 1024 / 1024 / 1024
                label = filename.split("_")[-1].split(".")[0]
                data[label] = gpu_usage

        names = list(data.keys())
        values = list(data.values())
        plt.figure(figsize=(10, 6)) 
        plt.bar(range(len(data)), values, tick_label=names)
        plt.xlabel("Model label")
        plt.ylabel("Maximum GPU memmory usage in GB")
        plt.grid()
        plt.savefig("{}/gpu_usage_plot.png".format(self.root_folder))
        plt.clf()



