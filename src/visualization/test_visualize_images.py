import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

import visualize_images

def test_show_image_batch_runs_without_error():
    # Create dummy images: 10 RGB images of size 32x32
    images = torch.rand(10, 3, 32, 32)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    classes = ['A', 'B']
    # Should not raise any error
    visualize_images.show_image_batch(images, 
                                      targets, 
                                      classes, 
                                      num_images=10, 
                                      rows=2, cols=5, 
                                      normalized=False)

def test_show_image_batch_with_normalization():
    images = torch.rand(4, 3, 32, 32)
    targets = torch.tensor([0, 1, 1, 0])
    classes = ['cat', 'dog']
    # Should not raise error with normalization
    visualize_images.show_image_batch(images, 
                                      targets, 
                                      classes, 
                                      num_images=4, 
                                      rows=2, 
                                      cols=2,
                                      normalized=True)

def test_show_image_batch_handles_less_images_than_requested():
    images = torch.rand(3, 3, 32, 32)
    targets = torch.tensor([0, 1, 0])
    classes = ['A', 'B']
    # Should not raise error if num_images > available images
    visualize_images.show_image_batch(images, 
                                      targets, 
                                      classes, 
                                      num_images=3, 
                                      rows=1, 
                                      cols=3, 
                                      normalized=False)

def test_show_image_batch_handles_empty_batch():
    images = torch.rand(0, 3, 32, 32)
    targets = torch.tensor([], dtype=torch.long)
    classes = ['A', 'B']
    # Should not raise error, but nothing will be plotted
    visualize_images.show_image_batch(images,
                                      targets,
                                      classes, 
                                      num_images=0, 
                                      rows=1, 
                                      cols=1, 
                                      normalized=False)