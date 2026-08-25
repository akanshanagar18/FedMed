"""
Inference helper
"""

from .predict import predict


def run_inference(model, image):

    """
    Complete inference workflow.
    """

    prediction = predict(model, image)

    return prediction