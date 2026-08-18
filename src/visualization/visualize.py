"""
Visualization utilities
"""

import matplotlib.pyplot as plt


def show_image(image):

    """
    Display MRI image.
    """

    plt.imshow(image, cmap="gray")
    plt.title("MRI Image")
    plt.axis("off")
    plt.show()


def show_mask(mask):

    """
    Display segmentation mask.
    """

    plt.imshow(mask)
    plt.title("Segmentation Mask")
    plt.axis("off")
    plt.show()