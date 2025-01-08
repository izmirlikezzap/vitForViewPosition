import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from data_utils import ClassificationDataset
from pathlib import Path
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PARAMS = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

# Updated number of classes
classes = ('Frontal','Lateral')
SEED_NUMBER = 4
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

model = ViT(pretrained=False)
model = model.to(DEVICE)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


writer = SummaryWriter(comment=f"_SEED_twoclass_NUMBER_{SEED_NUMBER}")
from tqdm import tqdm


print("current device : ", DEVICE)

BEST_ACCURACY = 0.0
for epoch in tqdm(range(70)):  # loop over the dataset multiple times


    running_loss = 0.0
    correct_predictions = 0
    wrong_predictions = []
    model = model.train()
    model = model.to(DEVICE)
    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.squeeze(dim=1).to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item()}")
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
            if test_predict is None:
                test_predict = predicted
                test_label = labels
            else:
                test_predict = torch.cat((test_predict, predicted))
                test_label = torch.cat((test_label, labels))
            cm, class_report, metric_labels, metric_results = evaluate(
                predictions=test_predict,
                labels=test_label,
                class_names=('Frontal','Lateral')
            )

        for label, result in zip(metric_labels, metric_results):
            if result is not None:
                if label == "Accuracy":
                    if BEST_ACCURACY < result:
                        BEST_ACCURACY = result
                        models_dir = "models"
                        if not os.path.exists(models_dir):
                            os.makedirs(models_dir)
                        torch.save(model.cpu(), f"models/best_model_twoclass_seed_{SEED_NUMBER}.pt")
                writer.add_scalar(f'{label}', result, epoch)

print('Finished Training')