import matplotlib.pyplot as plt
import numpy as np

from py.performances.performances_logger import PerformancesLogger


class GraphsPlotter:
    def __init__(self, root_folder, total_epochs, validation_interval):
        self.root_folder = root_folder
        self.total_epochs = total_epochs
        self.validation_interval = validation_interval

    def plot_performances(self):
        performances = np.load("{}/performances.npy".format(self.root_folder), allow_pickle=True)
        performances = performances.item()

        train_metrics = list(performances[list(performances.keys())[0]].keys())

        for metric in train_metrics:
            for dataset_type in performances.keys():
                performances = performances[dataset_type][metric]
                epochs = [epoch for epoch in range(self.total_epochs)]
                if performances.length != self.total_epochs:
                    epochs = [epoch for epoch in epochs if
                              PerformancesLogger.is_validation_epoch(epoch=epoch,
                                                                     validation_at=self.validation_interval)]
                plt.plot(epochs, performances, label=dataset_type)

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid()
            plt.savefig("{}/performance-plots/{}.png".format(self.root_folder, metric))
            plt.clf()

    def plot_execution_time(self):
        time_data = np.load("{}/execution_data.npy".format(self.root_folder), allow_pickle=True)
        time_data = time_data.item()

        for epoch, time in enumerate(time_data["time"]):
            plt.plot(epoch, time)
        plt.xlabel("Epoch")
        plt.ylabel("Execution time in minutes")
        plt.grid()
        plt.savefig("{}/time-plot.png".format(self.root_folder))
        plt.clf()
