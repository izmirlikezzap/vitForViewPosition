import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from data_utils import ClassificationDataset
from pathlib import Path
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate

PARAMS = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

# Updated number of classes
classes = ('AP', 'AP_horizontal', 'L', 'PA')
SEED_NUMBER = 5
torch.manual_seed(SEED_NUMBER)
torch.cuda.manual_seed(SEED_NUMBER)
torch.cuda.manual_seed_all(SEED_NUMBER)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Device is: {DEVICE}")

main = "/media/envisage/backup8tb/Padchest"
dataset_folder = Path(main) / "projectionClassificationImages"
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

model = ViT(pretrained=False, num_classes=len(classes))
model = model.to(DEVICE)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# TensorBoard log dosyalarını kaydetmek için bir SummaryWriter oluşturun
writer = SummaryWriter(comment=f"_SEED_NUMBER_{SEED_NUMBER}")
from tqdm import tqdm

BEST_ACCURACY = 0.0
for epoch in tqdm(range(70)):  # loop over the dataset multiple times

    running_loss = 0.0
    correct_predictions = 0  # Yanlış tahminleri saymak için yeni bir değişken ekleyin
    wrong_predictions = []  # Yanlış tahmin edilen örnekleri kaydetmek için bir liste oluşturun
    model = model.train()
    model = model.to(DEVICE)
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.squeeze().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Yanlış tahminleri hesaplayın
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
    writer.add_scalar('Training Loss', running_loss, epoch)

    running_loss = 0.0
    model = model.eval()
    test_predict = None
    test_label = None
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.squeeze().to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            if test_predict == None:
                test_predict = predicted
                test_label = labels
            else:
                test_predict = torch.cat((test_predict, predicted))
                test_label = torch.cat((test_label, labels))
            metric_labels, metric_results = evaluate(test_predict, test_label)

        for label, result in zip(metric_labels, metric_results):
            if label == "Accuracy":
                if BEST_ACCURACY < result:
                    BEST_ACCURACY = result
                    torch.save(model.cpu(), f"models/best_model_seed_{SEED_NUMBER}.pt")
            writer.add_scalar(f'{label}', result, epoch)

print('Finished Training')