"""
Overlay utilities
"""

import matplotlib.pyplot as plt


def overlay_prediction(image, mask):

    """
    Overlay prediction on MRI image.
    """

    plt.imshow(image, cmap="gray")
    plt.imshow(mask, alpha=0.4)
    plt.title("Prediction Overlay")
    plt.axis("off")
    plt.show()