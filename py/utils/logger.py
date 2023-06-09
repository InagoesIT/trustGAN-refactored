import os
import time

import numpy as np
import torch


class Logger:
    def __init__(self, state, ):
        self.state = state

    def log_execution_time(self, start_time, model_index):
        if model_index == 0:
            self.state.execution_data["time"] += [(time.time() - start_time) / 60]
        else:
            self.state.execution_data["time"][self.state.epoch] += (time.time() - start_time) / 60
            self.state.execution_data["time"][self.state.epoch] /= 2

    def load_logs(self, path_to_performances, execution_data_file_name, root_folder):
        if path_to_performances is None:
            return
        execution_data_path = execution_data_file_name.format(root_folder)
        if os.path.exists(execution_data_path):
            self.state.execution_data = np.load(execution_data_path, allow_pickle=True)
            self.state.execution_data = self.state.execution_data.item()
        if os.path.exists(path_to_performances):
            self.state.average_performances = np.load(path_to_performances, allow_pickle=True)
            self.state.average_performances = self.state.average_performances.item()

    def log_execution_data(self, root_folder, execution_data_file_name):
        self.state.execution_data["memory"] = [torch.cuda.max_memory_allocated(0)]
        np.save(execution_data_file_name.format(root_folder), self.state.execution_data)
