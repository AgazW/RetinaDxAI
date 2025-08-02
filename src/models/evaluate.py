import sys
import torch
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("src/models")
import train
sys.path.append("src")
from preprocess import preprocessing
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
    model = train.get_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(img_path, img_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    transform = preprocessing.get_transforms(img_size=img_size)
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def predict(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()
    return class_names[pred_idx]

def evaluate_model(model, val_loader):
    avg_val_loss, val_acc = train.evaluate_model(model, val_loader, device=device)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    plot_confusion_matrix(model, val_loader, class_names, device=device, normalize=False)
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, val_loader = train.get_dataloaders("data/processed/normalized_dataset_subset100.pt", batch_size=32)
    num_classes = len(Counter(val_loader.dataset.tensors[1].tolist()))

    # Load model
    model = load_model("models/Resnet_model_weights.pth", num_classes)

    # Evaluate on validation set
    evaluate_model(model, val_loader)

    # Predict a single image from validation set
    images, labels = next(iter(val_loader))
    single_image = images[0].unsqueeze(0).to(device)
    pred_class = predict(model, single_image)
    print(f"Predicted class for validation image: {pred_class}")

    # Predict a new image from file
    img_path = "data/external/data/Subset100/Healthy/Healthy559.jpg"
    img_tensor = preprocess_image(img_path)
    pred_class = predict(model, img_tensor)
    print(f"Predicted class for external image: {pred_class}")