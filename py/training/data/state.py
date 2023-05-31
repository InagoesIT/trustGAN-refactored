import torch


class State:
    def __init__(self, given_target_model=None, verbose=True, device_name=None, seed=42, model_label=""):
        self.given_target_model = given_target_model
        self.verbose = verbose
        self.epoch = 0
        self.model_label = model_label
        self.average_performances = None
        self.model_performances = None
        self.best_loss = None
        self.loss_gan = -1.0
        self.execution_data = {"time": []}
        self.seed = seed
        self.device = None
        self.set_device(device_name=device_name)
        self.initialize_performances()

    def set_device(self, device_name=None):
        if device_name is None:
            device_name = "cuda:0"

        if not torch.cuda.is_available():
            device_name = "cpu"

        device = torch.device(device_name)
        if self.verbose:
            print(f"Device = {device}, {device_name}")

        self.device = device
    
    def initialize_performances(self):
        self.average_performances = {"training": {}, "validation": {}}
        self.best_loss = float("inf")
        self.initialize_model_performances()

    def initialize_model_performances(self):
        self.model_performances = {"training": {}, "validation": {}}   
