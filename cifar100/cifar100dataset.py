import os
import pandas as pd
import torch
import random
import csv
import pickle
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMG_LABELS = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
ROOT_DIR = "clientdata"
CLIENTS = 100
ALPHA = 1

train_transform = transforms.Compose([
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

])

class cifar100Dataset(Dataset):
    def __init__(self, csv_file, transform=train_transform):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        x = self.annotations.iloc[index, :-1]
        x = torch.tensor(x)
        x = x/255
        x = x.reshape(3,32,32)
        y_label = torch.tensor(self.annotations.iloc[index, -1])
        if self.transform:
            image = self.transform(x)
        return image, y_label