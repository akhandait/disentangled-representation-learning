import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt

class CelebA(Dataset):
    def __init__(self, imgDirectory):
        self.imgPaths = []
        for img in os.listdir(imgDirectory):
            self.imgPaths.append(imgDirectory + img)

        self.transforms = transforms.Compose([transforms.Resize(64),
                                              transforms.CenterCrop(64),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), \
                                                  (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        img = Image.open(self.imgPaths[index])

        return self.transforms(img)

class Mnist(Dataset):
    def __init__(self, imgDirectory):
        self.imgPaths = []
        for img in os.listdir(imgDirectory):
            self.imgPaths.append(imgDirectory + img)

        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), \
                                                  (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        img = Image.open(self.imgPaths[index])

        return self.transforms(img)

# Visualize the training images.
if __name__ == '__main__':

    dataloader = DataLoader(CelebA(''), \
        batch_size=64, shuffle=True, num_workers=6) # Set the dataset class and path here.
    batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Sample Images')
    plt.imshow(np.transpose(vutils.make_grid(batch, padding=2, \
        normalize=True), (1, 2, 0)))
    plt.show()
