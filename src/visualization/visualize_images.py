import matplotlib.pyplot as plt
import numpy as np

def show_image_batch(images, 
                     targets, 
                     classes, 
                     num_images=10, 
                     rows=2, cols=5, 
                     mean=None, 
                     std=None, 
                     figsize=(15, 6),
                     normalized=True):
    """
    Display a batch of images with their class labels.

    Args:
        images (Tensor): Batch of images (shape: [batch_size, channels, height, width]).
        targets (Tensor): Batch of target labels (shape: [batch_size]).
        classes (list): List of class names.
        num_images (int): Number of images to display.
        rows (int): Number of rows in the plot grid.
        cols (int): Number of columns in the plot grid.
        mean (list or None): Mean for unnormalization. Defaults to ImageNet mean.
        std (list or None): Std for unnormalization. Defaults to ImageNet std.
        figsize (tuple): Figure size for the plot.
        normalized (bool): Whether the images are normalized and need to be unnormalized before display.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    def imshow(img):
        img = img.numpy().transpose((1, 2, 0))
        if normalized:
            img = std * img + mean  # unnormalize
            img = np.clip(img, 0, 1)
        else:
            img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.axis('off')

    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        imshow(images[i])
        plt.title(classes[targets[i]])
    plt.tight_layout()
    plt.show()