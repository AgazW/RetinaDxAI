import pytest
import torch
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import sys
sys.path.append("src")
from models import evaluate

def test_class_names_length():
    assert len(evaluate.class_names) == 10
    assert isinstance(evaluate.class_names[0], str)

def test_preprocess_image_with_pil():
    # Create a dummy PIL image
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    with patch("preprocess.preprocessing.get_transforms") as mock_transforms:
        mock_transform = MagicMock()
        mock_transform.return_value = torch.rand(3, 224, 224)
        mock_transforms.return_value = mock_transform
        tensor = evaluate.preprocess_image(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1  # Batch dimension

def test_preprocess_image_with_path(tmp_path):
    # Save a dummy image
    img_path = tmp_path / "test.jpg"
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    img.save(img_path)
    with patch("preprocess.preprocessing.get_transforms") as mock_transforms:
        mock_transform = MagicMock()
        mock_transform.return_value = torch.rand(3, 224, 224)
        mock_transforms.return_value = mock_transform
        tensor = evaluate.preprocess_image(str(img_path))
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1

def test_load_model(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self): super().__init__()
        def eval(self): return self
        def to(self, device): return self
        def load_state_dict(self, state): return self

    monkeypatch.setattr(evaluate.train, "get_resnet18", lambda num_classes, pretrained: DummyModel())
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: {})
    model = evaluate.load_model("dummy_path.pth", num_classes=10)
    assert isinstance(model, DummyModel)

def test_predict():
    # Mock model and tensor
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.1, 0.9] + [0.0]*8])
        def __call__(self, x): return self.forward(x)
    model = DummyModel()
    img_tensor = torch.rand(1, 3, 224, 224)
    # Patch class_names for 10 classes
    evaluate.class_names = [f"Class{i}" for i in range(10)]
    result = evaluate.predict(model, img_tensor)
    assert "Class1" in result
    assert "%" in result