import torch
from torch.utils.data import DataLoader, TensorDataset

# train.py

import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.

    Parameters:
    ----------
        num_classes (int): Number of output classes.
    """
    def __init__(self, num_classes):
        """
        Initializes the SimpleCNN model architecture.

        Parameters:
        ----------
            num_classes (int): Number of output classes.
        """
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 112 -> 56
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # Adjusted for image size 224x224
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        """
        Defines the forward pass of the SimpleCNN.

        Parameters:
        ----------
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(data_path, batch_size=32, val_split=0.2):
    """
    Loads data from a .pt or .pth file and returns PyTorch DataLoaders for training and validation.

    Parameters:
    ----------
        data_path (str): Path to the .pt or .pth file containing the data dictionary.
        batch_size (int, optional): Batch size for the DataLoaders. Defaults to 32.
        val_split (float, optional): Fraction of data to use for validation if not already split. Defaults to 0.2.

    Returns:
        tuple: (train_loader, val_loader) - DataLoaders for training and validation datasets.
    """
    data = torch.load(data_path)
    if 'X_train' in data and 'y_train' in data and 'X_val' in data and 'y_val' in data:
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
    else:
        X, y = data['X'], data['y']
        # Split into train and val
        num_samples = X.shape[0]
        indices = torch.randperm(num_samples)
        split = int(num_samples * (1 - val_split))
        train_idx, val_idx = indices[:split], indices[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Trains a PyTorch model using the provided DataLoaders.

    Parameters:
    ----------
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        None
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")


def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluates a PyTorch model on the provided DataLoader.

    Parameters:
    ----------
        model (nn.Module): The PyTorch model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation data.
        device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        float: Accuracy of the model on the provided data.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


