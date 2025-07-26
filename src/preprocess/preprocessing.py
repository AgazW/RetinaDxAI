
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os

def get_transforms(img_size=(224, 224)):
    """
    Create a torchvision transform pipeline for preprocessing images.

    Parameters:
    ----------
        img_size (tuple): Desired image size as (height, width).

    Returns:
        torchvision.transforms.Compose: Composed transform for image preprocessing.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # # Normalization values are ImageNet means/stds (see explanation below)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_image_dataloader(data_dir, batch_size=32, img_size=(224, 224), shuffle=True):
    """
    Create a DataLoader for images organized in subfolders of a directory.

    Parameters:
    ----------
        data_dir (str): Path to the root directory containing image subfolders.
        batch_size (int): Number of images per batch.
        img_size (tuple): Desired image size as (height, width).
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tuple: (DataLoader, list of class names, dict mapping class names to indices)
    """
    transform = get_transforms(img_size)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.classes, dataset.class_to_idx


def save_preprocessed_batches(dataloader, save_dir):
    """
    Saves each batch of images and targets from the dataloader as separate .pt files.
    
    Parameters:
    ----------
        dataloader: PyTorch DataLoader yielding (images, targets)
        save_dir: Directory to save .pt files
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, (images, targets) in enumerate(dataloader):
        batch_path = os.path.join(save_dir, f"batch_{i:04d}.pt")
        torch.save({'images': images, 'targets': targets}, batch_path)



def save_entire_dataset(dataloader, save_path):
    """
    Saves the entire dataset as a single .pt file containing all images and targets tensors.
    
    Parameters:
    ----------
        dataloader: PyTorch DataLoader yielding (images, targets)
        save_path: Path to save the .pt file
    """
    all_images = []
    all_targets = []
    for images, targets in dataloader:
        all_images.append(images)
        all_targets.append(targets)
    images_tensor = torch.cat(all_images, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    torch.save({'images': images_tensor, 'targets': targets_tensor}, save_path)