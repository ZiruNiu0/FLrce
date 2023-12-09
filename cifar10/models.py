import torch.nn as nn
from PIL import Image
import torch
import numpy as np

NUM_CLASSES = 10
CHANNELS = 3

class CNN(nn.Module):
    def __init__(self, in_channels, outputs=10, rate=1.0) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,max(1, int(6*rate)),kernel_size=(3,3),padding=1)
        self.bn1 = nn.BatchNorm2d(max(1, int(6*rate)))
        self.conv2 = nn.Conv2d(max(1, int(6*rate)),max(1, int(16*rate)),kernel_size=(3,3),padding=1)
        self.bn2 = nn.BatchNorm2d(max(1, int(16*rate)))
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(max(1, int(16*rate))*8*8, max(1, int(120*rate)),bias=False)
        self.fc2 = nn.Linear(max(1, int(120*rate)), max(1, int(84*rate)), bias=False)
        self.fc = nn.Linear(max(1, int(84*rate)), outputs)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc(x)
        return x

def add_group_lasso(model:CNN):
    filter_lasso_1 = model.conv1.weight.pow(2).sum(dim=(3,2)).pow(0.5).sum()
    channel_lasso_1 = model.conv1.weight.pow(2).sum(dim=(3,2,1)).pow(0.5).sum()
    filter_lasso_2 = model.conv2.weight.pow(2).sum(dim=(3,2)).pow(0.5).sum()
    channel_lasso_2 = model.conv2.weight.pow(2).sum(dim=(3,2,1)).pow(0.5).sum()
    fc1_weight_lasso = model.fc1.weight.pow(2).sum(dim=(1)).pow(0.5).sum()
    fc2_weight_lasso = model.fc2.weight.pow(2).sum(dim=(1)).pow(0.5).sum()
    return filter_lasso_1 + channel_lasso_1 + filter_lasso_2 + channel_lasso_2 + fc1_weight_lasso + fc2_weight_lasso
