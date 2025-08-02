import sys
import torch
from collections import Counter
import matplotlib.pyplot as plt

sys.path.append("src/models")
import train
sys.path.append("src")
from visualization.plot_performance import plot_confusion_matrix

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

# Class names
class_names = [
    'Central Serous Chorioretinopathy',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

def load_model(weights_path, num_classes):
    """
    Loads a ResNet18 model with the specified weights and number of classes.

    Parameters
    ----------
    weights_path : str
        Path to the model weights file.
    num_classes : int
        Number of output classes for the model.

    Returns
    -------
    model : torch.nn.Module
        The loaded and prepared model.
    """
    model = train.get_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(img, img_size=(224, 224)):
    """
    Preprocesses an image for model prediction.

    Parameters
    ----------
    img : str or PIL.Image.Image
        File path to the image or a PIL Image object.
    img_size : tuple, optional
        Desired image size (default is (224, 224)).

    Returns
    -------
    img_tensor : torch.Tensor
        Preprocessed image tensor ready for model input.
    """
    from PIL import Image
    from preprocess import preprocessing
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    transform = preprocessing.get_transforms(img_size=img_size)
    img_tensor = transform(img).unsqueeze(0).to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    return img_tensor

def predict(model, img_tensor):
    """
    Predicts the class of a given image tensor using the provided model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model for prediction.
    img_tensor : torch.Tensor
        Preprocessed image tensor.

    Returns
    -------
    tuple
        Predicted class name (str) and confidence score (float).
    """
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item() * 100  # Convert to percentage
    return f"{class_names[pred_idx]} ({confidence:.2f}%)"
