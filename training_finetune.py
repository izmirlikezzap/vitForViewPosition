import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from data_utils import ClassificationDataset
from pathlib import Path
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate
from tqdm import tqdm

PARAMS = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

# Updated number of classes
classes = ('Frontal', 'Lateral')
SEED_NUMBER = 3
torch.manual_seed(SEED_NUMBER)
torch.cuda.manual_seed(SEED_NUMBER)
torch.cuda.manual_seed_all(SEED_NUMBER)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Device is: {DEVICE}")

main = "/media/envisage/backup8tb/Padchest"
dataset_folder = Path(main) / "projectionClassificationImagesTwoClass"
train_folder = dataset_folder / "train"
test_folder = dataset_folder / "test"
val_folder = dataset_folder / "validation"

train_dataset = ClassificationDataset(folder=train_folder)
train_loader = DataLoader(train_dataset, **PARAMS)

val_dataset = ClassificationDataset(folder=val_folder)
val_loader = DataLoader(val_dataset, **PARAMS)

test_dataset = ClassificationDataset(folder=test_folder)
test_loader = DataLoader(test_dataset, **PARAMS)

from vit import ViT

# Load pre-trained ViT model
model = ViT(pretrained=False)
loaded_model = torch.load("models/best_model_twoclass_seed_3.pt")  # Load complete model
torch.save(loaded_model.state_dict(), "models/best_model_twoclass_seed_3_state_dict.pt")  # Save state_dict

state_dict = torch.load("models/best_model_twoclass_seed_3_state_dict.pt")
model.load_state_dict(state_dict)
model = model.to(DEVICE)

print("Pre-trained model loaded successfully!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# TensorBoard
writer = SummaryWriter(comment=f"_SEED_NUMBER_{SEED_NUMBER}_twoClass_finetune")

BEST_ACCURACY = 0.0

# Training loop
for epoch in tqdm(range(70)):  # Train for 70 epochs
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()


        ##print("labels:", labels)
        # Forward pass
        outputs = model(inputs)
        labels = labels.squeeze(dim = 1)
        #print("shape of labels", labels.shape)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = 100.0 * correct_train / total_train
    print(f"Epoch {epoch + 1}, Training Loss: {running_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    writer.add_scalar('Training Loss', running_loss, epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)

    # Validation
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = 100.0 * correct_val / total_val
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    # Save the best model
    if val_accuracy > BEST_ACCURACY:
        BEST_ACCURACY = val_accuracy
        torch.save(model.cpu(), f"models/best_twoClass_model_seed_{SEED_NUMBER}_finetune_3.pt")
        model = model.to(DEVICE)  # Return model to GPU

print("Finished Training")
