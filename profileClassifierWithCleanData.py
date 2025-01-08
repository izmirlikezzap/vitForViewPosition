import os
import torch
import random
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

# Paths
main = "/media/envisage/backup8tb/Padchest"
dataset_folder = Path(main) / "projectionClassificationImagesTwoClass"
train_folder = dataset_folder / "train"
test_folder = dataset_folder / "test"
val_folder = dataset_folder / "validation"

# Hyperparameters
batch_size = 64  # Adjust based on 24GB GPU
num_epochs = 10
learning_rate = 0.001
num_seeds = 5
models_to_train = [
    models.resnet18,
    models.mobilenet_v2,
    models.alexnet,
    models.densenet121,
    models.vgg16,
    models.inception_v3,
    models.efficientnet_b0,
    models.squeezenet1_0,
    models.shufflenet_v2_x1_0,
    models.resnext50_32x4d
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets and DataLoaders
train_dataset = datasets.ImageFolder(train_folder, transform=transform)
test_dataset = datasets.ImageFolder(test_folder, transform=transform)
val_dataset = datasets.ImageFolder(val_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training and Evaluation Functions
def train_model(model, criterion, optimizer, dataloader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

def evaluate_model(model, criterion, dataloader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

# Training Loop
for model_func in models_to_train:
    for seed in range(num_seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        print(f"\nTraining {model_func.__name__} with seed {seed}...")

        model = model_func(pretrained=True)
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif hasattr(model, 'classifier'):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_acc = 0.0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, train_acc = train_model(model, criterion, optimizer, train_loader)
            val_loss, val_acc = evaluate_model(model, criterion, val_loader)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"best_{model_func.__name__}_seed{seed}.pth")

        print(f"Best Validation Accuracy for {model_func.__name__} with seed {seed}: {best_val_acc:.4f}")
