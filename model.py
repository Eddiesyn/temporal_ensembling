# coding=utf-8

import torch
import torch.nn as nn
import torchvision

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape, std):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size, ) + input_shape
        self.std = std
        self.noise = torch.zeros(self.shape).cuda()

    def forward(self, x):
        self.noise.normal_(mean=0, std=self.std)

        return x + self.noise

class HandyModel(nn.Module):
    def __init__(self, batch_size=32, input_shape=(1, 28, 28), std=0.05, p=0.5):
        super(HandyModel, self).__init__()
        self.std = std
        self.p = p
        self.gn = GaussianNoise(batch_size, input_shape, self.std)
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        if self.training:
            x = self.gn(x)

        # first block
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(x)
        x = self.pool(x)

        # second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(x)
        x = self.pool(x)

        # classifier
        x = self.fc(x)

        return x