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
        # print(self.noise.shape)

        return x + self.noise

class HandyModel(nn.Module):
    def __init__(self, batch_size, std, input_shape=(1, 28, 28), p=0.5, fm1=16, fm2=32):
        super(HandyModel, self).__init__()
        self.std = std
        self.p = p
        self.fm1 = fm1
        self.fm2 = fm2
        self.gn = GaussianNoise(batch_size, input_shape, self.std)
        self.conv1 = nn.Conv2d(1, self.fm1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.fm1, self.fm2, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.p)
        self.fc = nn.Linear(self.fm2 * 7 * 7, 10)
        self.bn1 = nn.BatchNorm2d(self.fm1)
        self.bn2 = nn.BatchNorm2d(self.fm2)

    def forward(self, x):
        if self.training:
            # print(x.shape)
            x = self.gn(x)

        # first block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        # second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)

        # classifier
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)

        return x
