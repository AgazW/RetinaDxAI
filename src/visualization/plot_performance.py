import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch

def plot_results(results, epochs):
    """
    Plot training and validation loss, and validation accuracy for each model.

    ----
    Parameters
    ----------
    results : dict
        Dictionary where each key is a model name and each value is a dict with keys
        'train_loss', 'val_loss', and 'val_acc', each mapping to a list of values per epoch.
    epochs : iterable
        Iterable of epoch numbers.

    Returns
    -------
    None
        This function displays the plots and does not return any value.

    The function creates two subplots:
        - Left: Training and validation loss vs. epochs.
        - Right: Validation accuracy vs. epochs.
    Both axes use whole number ticks for clarity.
    """

    plt.figure(figsize=(12,5), dpi = 200)

    plt.subplot(1,2,1)
    for name in results:
        plt.plot(list(epochs), results[name]['train_loss'], label=f'{name} Train Loss')
        plt.plot(list(epochs), results[name]['val_loss'], '--', label=f'{name} Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.xticks([int(x) for x in epochs])
    # Set y-ticks as whole numbers within the range of loss values
    min_loss = min([min(results[name]['train_loss'] + results[name]['val_loss']) for name in results])
    max_loss = max([max(results[name]['train_loss'] + results[name]['val_loss']) for name in results])
    plt.yticks(range(math.floor(min_loss), math.ceil(max_loss)+1))

    plt.subplot(1,2,2)
    for name in results:
        plt.plot(list(epochs), results[name]['val_acc'], label=f'{name} Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend()
    plt.xticks([int(x) for x in epochs])
    # Set y-ticks as whole numbers within the range of accuracy values
    min_acc = min([min(results[name]['val_acc']) for name in results])
    max_acc = max([max(results[name]['val_acc']) for name in results])
    plt.yticks(range(math.floor(min_acc), math.ceil(max_acc)+1))

    plt.tight_layout()
    plt.show() 



def plot_confusion_matrix(model, data_loader, class_names, device='cpu', normalize=False, figsize=(6,5)):
    """
    Plots the confusion matrix for a given model and dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the dataset to evaluate.
    class_names : list
        List of class names (strings).
    device : str
        Device to run the model on ('cpu' or 'cuda').
    normalize : bool
        If True, normalize the confusion matrix.
    figsize : tuple
        Size of the plot.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds, normalize='true' if normalize else None)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.show()