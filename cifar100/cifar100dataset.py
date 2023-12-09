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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
        
def create_iid_datasets(seed=2023, clients=CLIENTS, aplha=ALPHA):
    random.seed(seed)
    csvfile = open('cifar10_global_train.csv', 'r')
    datareader = csv.reader(csvfile)
    distributed_data = {}
    for i in range(clients):
        distributed_data[i] = []
    for row in datareader:
        client = random.randint(0, clients-1)
        distributed_data[client].append(row)
    for c in range(clients):
        dataname = "cifar10_client_" + str(c) + "_ALPHA_" + str(aplha) + ".csv"
        with open(dataname, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in distributed_data[c]:
                writer.writerow(row)

def create_unbalanced_datasets(alpha, clients=CLIENTS):
    csvfile = open('cifar100_global_train.csv', 'r')
    datareader = csv.reader(csvfile)
    distributed_data = {}
    alphas = np.ones(clients) * alpha
    dirichlet_variables = np.random.dirichlet(alphas, size=1)
    candidates = list(range(clients))
    for i in range(clients):
        distributed_data[i] = []
    for row in datareader:
        client = random.choices(candidates,weights=list(dirichlet_variables.flatten()), k=1)[0]
        while len(distributed_data[client]) >= 1200:
            client = random.choice(candidates) 
        distributed_data[client].append(row)
    for c in range(clients):
        print(f"Writing file for Client {c}...")
        dataname = "clientdata/cifar100_client_" + str(c) + "_ALPHA_" + str(alpha) + ".csv"
        with open(dataname, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in distributed_data[c]:
                writer.writerow(row)
        
def create_global_dataset(Root_dir=ROOT_DIR):
    trainfiles = ['cifar-100-python/train']
    testfile = 'cifar-100-python/test'
    traindata, testdata = "cifar100_global_train.csv", "cifar100_global_test.csv"
    pixels, labels = [], []
    for trainfile in trainfiles:
        data_batch = unpickle(trainfile)
        for p, l in zip(data_batch[b'data'], data_batch[b'coarse_labels']):
            labels.append(l)
            pixels.append(p)
    with open(traindata, 'w', newline='') as tf:
        writer = csv.writer(tf)
        for sample, Class in zip(pixels, labels):
            sample = np.append(sample, Class)
            writer.writerow(sample)
    pixels, labels = [], []
    data_batch = unpickle(testfile)
    for p, l in zip(data_batch[b'data'], data_batch[b'fine_labels']):
        labels.append(l)
        pixels.append(p)
    with open(testdata, 'w', newline='') as tf2:
        writer = csv.writer(tf2)
        for sample, Class in zip(pixels, labels):
            sample = np.append(sample, Class)
            writer.writerow(sample)

def get_dataset_bad(cid):
    if cid % 100 < 20:
        return 900, 0
    else:
        return 400, 0

if __name__ == "__main__":
  alpha = 0.1
  create_unbalanced_datasets(alpha)
  #create_global_dataset()