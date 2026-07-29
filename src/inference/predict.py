"""
Prediction Pipeline
"""

import torch


def predict(model, image):

    """
    Run inference using trained model.
    """

    model.eval()

    with torch.no_grad():

        prediction = model(image)

    return prediction