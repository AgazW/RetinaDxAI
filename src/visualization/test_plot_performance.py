import pytest
import torch
import numpy as np
from unittest import mock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

import plot_performance

def test_plot_results_runs_without_error():
    # Dummy results for two models, three epochs
    results = {
        'SimpleCNN': {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [1.1, 0.9, 0.7],
            'val_acc': [0.5, 0.6, 0.7]
        },
        'ResNet18': {
            'train_loss': [0.9, 0.7, 0.5],
            'val_loss': [1.0, 0.8, 0.6],
            'val_acc': [0.55, 0.65, 0.75]
        }
    }
    epochs = range(1, 4)
    # Should not raise any error
    plot_performance.plot_results(results, epochs)

def test_plot_confusion_matrix_basic():
    # Create a dummy model that outputs fixed logits
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Always predict class 0 for batch size
            batch_size = x.shape[0]
            logits = torch.zeros((batch_size, 2))
            logits[:, 0] = 1.0
            return logits

    # Dummy data loader: 6 samples, 2 classes
    images = torch.randn(6, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    dataset = torch.utils.data.TensorDataset(images, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    class_names = ['A', 'B']

    # Should not raise any error
    plot_performance.plot_confusion_matrix(DummyModel(), data_loader, class_names, device='cpu', normalize=False)
    plot_performance.plot_confusion_matrix(DummyModel(), data_loader, class_names, device='cpu', normalize=True)

def test_plot_confusion_matrix_handles_empty():
    # Empty data loader
    images = torch.randn(0, 3, 32, 32)
    labels = torch.tensor([], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(images, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    class_names = ['A', 'B']
    # Should not raise error even if confusion matrix is empty
    plot_performance.plot_confusion_matrix(torch.nn.Identity(), data_loader, class_names, device='cpu', normalize=False)

def test_plot_confusion_matrix_class_names_length():
    # If class_names length does not match number of classes, should raise error
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = torch.utils.data.TensorDataset(images, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    wrong_class_names = ['A']  # Should be 2
    with pytest.raises(Exception):
        plot_performance.plot_confusion_matrix(torch.nn.Identity(), data_loader, wrong_class_names, device='cpu', normalize=False)