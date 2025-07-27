import matplotlib.pyplot as plt
import math

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