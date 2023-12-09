import os
import pandas as pd
import torch
import random
import csv
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMG_LABELS = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
ROOT_DIR = "CIFAR-10-images/train/"
CLIENTS = 100
ALPHA = 1

train_transform = transforms.Compose([
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

])

class cifar10Dataset(Dataset):
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

def create_unbalanced_datasets(alpha, clients=CLIENTS):
    csvfile = open('cifar10_global_train.csv', 'r')
    datareader = csv.reader(csvfile)
    distributed_data = {}
    alphas = np.ones(clients) * alpha
    dirichlet_variables = np.random.dirichlet(alphas, size=1)
    candidates = list(range(clients))
    for i in range(clients):
        distributed_data[i] = []
    for row in datareader:
        client = random.choices(candidates,weights=list(dirichlet_variables.flatten()), k=1)[0]
        distributed_data[client].append(row)
    for c in range(clients):
        print(f"Writing file for Client {c}...")
        dataname = "clientdata/cifar10_client_" + str(c) + "_ALPHA_" + str(alpha) + ".csv"
        with open(dataname, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in distributed_data[c]:
                writer.writerow(row)

if __name__ == "__main__":
  alpha = 1.0
  create_unbalanced_datasets(alpha)