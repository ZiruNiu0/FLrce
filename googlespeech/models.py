import torch.nn as nn

CHANNELS = 1
CLASSES = 35

class CNN(nn.Module):
    def __init__(self, in_channels=CHANNELS, outputs=CLASSES, rate=1.0) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, max(1, int(50*rate)), kernel_size=2, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(max(1, int(50*rate)))
        self.conv2 = nn.Conv2d(max(1, int(50*rate)), max(1, int(50*rate)), kernel_size=2, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(max(1, int(50*rate)))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU()
        self.fc = nn.Linear(max(1, int(50*rate))*3*3, outputs)
    
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
        x = self.fc(x)
        return x

def add_group_lasso(model:CNN):
    filter_lasso_1 = model.conv1.weight.pow(2).sum(dim=(3,2)).pow(0.5).sum()
    channel_lasso_1 = model.conv1.weight.pow(2).sum(dim=(3,2,1)).pow(0.5).sum()
    filter_lasso_2 = model.conv2.weight.pow(2).sum(dim=(3,2)).pow(0.5).sum()
    channel_lasso_2 = model.conv2.weight.pow(2).sum(dim=(3,2,1)).pow(0.5).sum()
    return filter_lasso_1 + channel_lasso_1 + filter_lasso_2 + channel_lasso_2
