"""
Prediction Pipeline
FedMed - Inference Module
"""

import torch


class Predictor:
    """
    Handles model inference.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """
        Run inference on a single MRI volume.
        """

        image = image.to(self.device)

        prediction = self.model(image)

        return prediction


def predict(model, image):
    """
    Simple prediction function for compatibility.
    """

    model.eval()

    with torch.no_grad():
        prediction = model(image)

    return prediction