# !/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, ntype):
        super(Net, self).__init__()
        self.ntype = ntype

        if self.ntype == "Linear":
            self.backbone = nn.Linear(9, 128)
            self.bn = nn.BatchNorm1d(128)
        elif self.ntype == "Conv":
            self.backbone = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=0)
            self.bn = nn.BatchNorm2d(128)
        else:
            raise ValueError("self.ntype can only take \"Linear\" and \"Conv\"")

        
        self.fc1 = nn.Linear(128, 64)
        self. bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self. bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

        
    def forward(self, x):
        # 3*3*1 --> 1*1*128
        x = self.backbone(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, 128)
        # 1*128 --> 1*64
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # 1*64 --> 1*64
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # 1*64 --> 1*1
        x = self.fc3(x)
        # x = F.sigmoid(x)
        x = x.squeeze(-1)
        return x