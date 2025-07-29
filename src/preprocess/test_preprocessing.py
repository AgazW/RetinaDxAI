import os
import shutil
import tempfile
import torch
import pytest
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import sys

import preprocessing

@pytest.fixture
def temp_image_folder():
    # Create a temporary directory with subfolders and dummy images
    temp_dir = tempfile.mkdtemp()
    class_names = ['classA', 'classB']
    for cls in class_names:
        cls_dir = os.path.join(temp_dir, cls)
        os.makedirs(cls_dir)
        # Create 2 dummy images per class
        for i in range(2):
            img = Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))
            img.save(os.path.join(cls_dir, f"{cls}_{i}.png"))
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_get_transforms():
    tf = preprocessing.get_transforms((128,128))
    assert isinstance(tf, transforms.Compose)
    # Check that Resize is in the pipeline
    assert any(isinstance(t, transforms.Resize) for t in tf.transforms)

def test_get_image_dataloader(temp_image_folder):
    dataloader, classes, class_to_idx = preprocessing.get_image_dataloader(
        temp_image_folder, batch_size=2, img_size=(64,64), shuffle=False
    )
    assert isinstance(dataloader, DataLoader)
    assert set(classes) == {'classA', 'classB'}
    assert set(class_to_idx.keys()) == {'classA', 'classB'}
    batch = next(iter(dataloader))
    images, targets = batch
    assert images.shape[1:] == (3, 64, 64)  # Channels, H, W
    assert targets.shape[0] == 2

def test_save_preprocessed_batches(temp_image_folder):
    dataloader, _, _ = preprocessing.get_image_dataloader(
        temp_image_folder, batch_size=2, img_size=(32,32), shuffle=False
    )
    with tempfile.TemporaryDirectory() as save_dir:
        preprocessing.save_preprocessed_batches(dataloader, save_dir)
        files = os.listdir(save_dir)
        assert any(f.startswith("batch_") and f.endswith(".pt") for f in files)
        # Check file content
        data = torch.load(os.path.join(save_dir, files[0]))
        assert 'images' in data and 'targets' in data

def test_save_entire_dataset(temp_image_folder):
    dataloader, _, _ = preprocessing.get_image_dataloader(
        temp_image_folder, batch_size=2, img_size=(16,16), shuffle=False
    )
    with tempfile.TemporaryDirectory() as save_dir:
        save_path = os.path.join(save_dir, "dataset.pt")
        preprocessing.save_entire_dataset(dataloader, save_path)
        assert os.path.exists(save_path)
        data = torch.load(save_path)
        assert 'images' in data and 'targets' in data
        assert data['images'].shape[1:] == (3, 16, 16)
