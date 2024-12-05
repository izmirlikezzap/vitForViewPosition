from torch import load, LongTensor, zeros
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


IMAGE_SIZE = 224
PARAMS = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 1}

class ClassificationDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, folder):
        'Initialization'
        self.train_folder = folder
        self.train_paths = list(folder.glob('**/*.png'))
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path = self.train_paths[index]
        image = Image.open(path)
        image = image.convert('RGB')
        X = self.transform(image)

        # Determine the label based on the folder name
        if "AP" in path.parts[-2]:
            label = LongTensor([0])
        elif "AP_horizontal" in path.parts[-2]:
            label = LongTensor([1])
        elif "L" in path.parts[-2]:
            label = LongTensor([2])
        elif "PA" in path.parts[-2]:
            label = LongTensor([3])
        else:
            raise ValueError("Unknown class label")

        return X, label


