import numpy as np
import torch


class Inference:
    def __init__(self, target_model, modifier, device, nr_classes):
        self.target_model = target_model
        self.modifier = modifier
        self.device = device
        self.nr_classes = nr_classes

    @torch.inference_mode()
    def get_inference_results(self, loader, score_type="MCP"):
        self.target_model.eval()

        softmax = torch.nn.Softmax(dim=1)
        truths = []
        predictions = []
        scores = []

        for data in loader:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            inputs, labels = self.modifier((inputs, labels))

            outputs = softmax(self.target_model(inputs))
            _, tmp_predictions = torch.max(outputs, 1)
            outputs, _ = torch.sort(outputs, dim=1)
            if score_type == "MCP":
                tmp_score = outputs[:, -1]
            elif score_type == "Diff2MCP":
                tmp_score = outputs[:, -1] - outputs[:, -2]
            else:
                return None

            normalizing_factor = 1 / self.nr_classes
            tmp_score = (tmp_score - normalizing_factor) / (1 - normalizing_factor)
            _, tmp_truth = torch.max(labels, 1)

            truths.append(tmp_truth.detach().cpu())
            predictions.append(tmp_predictions.detach().cpu())
            scores.append(tmp_score.detach().cpu())

        truths = torch.cat(truths)
        predictions = torch.cat(predictions)
        score = torch.cat(scores)

        self.target_model.train()

        return truths, predictions, score
