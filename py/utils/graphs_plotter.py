import os

import matplotlib.pyplot as plt
import numpy as np

from py.performances.performances_logger import PerformancesLogger


class GraphsPlotter:
    def __init__(self, root_folder, total_epochs, validation_interval, file_name_of_performances):
        self.root_folder = root_folder
        self.total_epochs = total_epochs
        self.validation_interval = validation_interval
        self.average_performances = np.load("{}/{}".format(self.root_folder, file_name_of_performances), allow_pickle=True)
        self.average_performances = self.average_performances.item()
        self.train_metrics = list(self.average_performances[list(self.average_performances.keys())[0]].keys()) 
    
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
