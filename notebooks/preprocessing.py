
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(img_size=(224, 224)):
    """
    Create a torchvision transform pipeline for preprocessing images.

    Args:
        img_size (tuple): Desired image size as (height, width).

    Returns:
        torchvision.transforms.Compose: Composed transform for image preprocessing.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # Normalization values are ImageNet means/stds (see explanation below)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_image_dataloader(data_dir, batch_size=32, img_size=(224, 224), shuffle=True):
    """
    Create a DataLoader for images organized in subfolders of a directory.

    Args:
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