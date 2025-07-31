import torch
from torch.utils.data import DataLoader, TensorDataset
import train

class DummyModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = torch.nn.Linear(3*8*8, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

def make_dummy_loader(num_samples=20, num_classes=2, img_shape=(3,8,8)):
    images = torch.randn(num_samples, *img_shape)
    labels = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)

def test_get_dataloaders_split(tmp_path):
    # Save dummy data
    images = torch.randn(20, 3, 8, 8)
    targets = torch.randint(0, 2, (20,))
    torch.save({'images': images, 'targets': targets}, tmp_path/"dummy.pt")
    train_loader, val_loader = train.get_dataloaders(str(tmp_path/"dummy.pt"), batch_size=4, val_split=0.25)
    # Check loaders
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert len(train_loader.dataset) + len(val_loader.dataset) == 20

def test_get_dataloaders_pre_split(tmp_path):
    # Save dummy pre-split data
    X_train = torch.randn(10, 3, 8, 8)
    y_train = torch.randint(0, 2, (10,))
    X_val = torch.randn(5, 3, 8, 8)
    y_val = torch.randint(0, 2, (5,))
    torch.save({'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}, tmp_path/"dummy2.pt")
    train_loader, val_loader = train.get_dataloaders(str(tmp_path/"dummy2.pt"), batch_size=2)
    assert len(train_loader.dataset) == 10
    assert len(val_loader.dataset) == 5

def test_evaluate_model_runs():
    loader = make_dummy_loader()
    model = DummyModel(num_classes=2)
    avg_loss, acc = train.evaluate_model(model, loader, device='cpu')
    assert isinstance(avg_loss, float)
    assert 0.0 <= acc <= 1.0

def test_train_model_runs():
    train_loader = make_dummy_loader()
    val_loader = make_dummy_loader()
    model = DummyModel(num_classes=2)
    train_losses, val_losses, val_accuracies = train.train_model(model, train_loader, val_loader, epochs=2, lr=0.01, device='cpu')
    assert len(train_losses) == 2
    assert len(val_losses) == 2
    assert len(val_accuracies) == 2
    assert all(isinstance(x, float) for x in train_losses + val_losses)
    assert all(0.0 <= x <= 1.0 for x in val_accuracies)