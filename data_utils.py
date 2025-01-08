from torch import load, LongTensor, zeros
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

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

        # Debug: Check if any images are found
        if len(self.train_paths) == 0:
            raise ValueError(f"No images found in folder: {folder}")

        print(f"Total images found: {len(self.train_paths)}")

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

        # Debug: Check file existence and readability
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            image = Image.open(path)
            image = image.convert('RGB')



            X = self.transform(image)



            if "Frontal" in path.parts[-2]:
                label = LongTensor([0])
            elif "Lateral" in path.parts[-2]:
                label = LongTensor([1])
            else:
                raise ValueError(f"Unknown class label for image: {path}")

            return X, label

        except Exception as e:
            print(f"Error processing image {path}: {e}")
            raise