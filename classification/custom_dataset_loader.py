import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from skimage import io

class StateFarmDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,1], self.annotations.iloc[index,2])
        image = io.imread(img_path)
        y_label_str = self.annotations.iloc[index,1]
        y_label = torch.tensor(int(y_label_str[-1]))

        if self.transform:
            image = torchvision.transforms.ToPILImage()(image)
            image = torchvision.transforms.Resize(192)(image)
            image = torchvision.transforms.ToTensor()(image)

        return (image, y_label)

