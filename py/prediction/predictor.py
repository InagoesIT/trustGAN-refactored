import numpy as np
import torch


class Predictor:
    def __init__(self, networks_data, modifier, device):
        self.networks_data = networks_data
        self.modifier = modifier
        self.device = device

    @torch.inference_mode()
    def get_predictions(self, loader, score_type="MCP"):
        if type(loader) == str:
            loader = getattr(self, loader)

        self.networks_data.target_model.eval()

        softmax = torch.nn.Softmax(dim=1)
        truth = []
        predictions = []
        score = []

        for data in loader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs, labels = self.modifier((inputs, labels))

            outputs = softmax(self.networks_data.target_model(inputs))
            _, tmp_predictions = torch.max(outputs, 1)
            outputs, _ = torch.sort(outputs, dim=1)
            if score_type == "MCP":
                tmp_score = outputs[:, -1]
            elif score_type == "Diff2MCP":
                tmp_score = outputs[:, -1] - outputs[:, -2]
            _, tmp_truth = torch.max(labels, 1)

            truth += [tmp_truth.detach().cpu().numpy()]
            predictions += [tmp_predictions.detach().cpu().numpy()]
            score += [tmp_score.detach().cpu().numpy()]

        truth = np.hstack(truth)
        predictions = np.hstack(predictions)
        score = np.hstack(score)

        self.networks_data.target_model.run()

        return truth, predictions, score
