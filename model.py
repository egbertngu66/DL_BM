# !/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import Module
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(1)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                           


class Net(Module.Model):
    def __init__(self, ntype, inshape=None):
        super(Net, self).__init__()
        self.ntype = ntype
        if self.ntype == "Linear":
            self.backbone = Module.Linear(9, 128)
            self.bn = Module.BatchNorm(2, 128)
        elif self.ntype == "Conv":
            self.backbone = Module.Conv2d(inshape, 128, ksize=3, stride=1, padding=0)
            self.bn = Module.BatchNorm(4, 128)
        else:
            raise ValueError("self.ntype can only take \"Linear\" and \"Conv\"")

        self.relu = Module.Relu()

        self.fc1 = Module.Linear(128, 64)
        self.bn1 = Module.BatchNorm(2, 64)
        self.relu1 = Module.Relu()

        self.fc2 = Module.Linear(64, 64)
        self. bn2 = Module.BatchNorm(2, 64)
        self.relu2 = Module.Relu()

        self.fc3 = Module.Linear(64, 1)
        self.sigmoid = Module.Sigmoid()
        
        self.layers_dict = {
            "backbone": self.backbone,
            "bn": self.bn,
            "relu": self.relu,
            "fc1": self.fc1,
            "bn1": self.bn1,
            "relu1": self.relu1,
            "fc2": self.fc2,
            "bn2": self.bn2,
            "relu2": self.relu2,
            "fc3": self.fc3,
            "sigmoid": self.sigmoid
        }


        self.backbone_pt = nn.Linear(9, 128, bias=False)
        self.fc1_pt = nn.Linear(128, 64, bias=False)
        self.fc2_pt = nn.Linear(64, 64, bias=False)
        self.fc3_pt = nn.Linear(64, 1, bias=False)

        
    def forward(self, x):
        x = self.backbone.forward(x)
        x = self.bn.forward(x)
        x = self.relu.forward(x)

        x = x.reshape(-1, 128)
        # 1*128 --> 1*64
        x = self.fc1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)

        # 1*64 --> 1*64
        x = self.fc2.forward(x)
        x = self.bn2.forward(x)
        x = self.relu2.forward(x)

        # 1*64 --> 1*1
        x = self.fc3.forward(x)
        x = self.sigmoid.forward(x)
        x = x.reshape(-1)
        return x
    

class NetTorch(nn.Module):
    def __init__(self, ntype):
        super(NetTorch, self).__init__()
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
        # print("backbone_pt.weight", self.backbone.weight)
        x = self.backbone(x)
        # x = self.bn(x)
        x = F.relu(x)
        x = x.view(-1, 128)
        # 1*128 --> 1*64
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        # 1*64 --> 1*64
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # 1*64 --> 1*1
        x = self.fc3(x)
        # x = torch.sigmoid(x)
        x = x.squeeze(-1)
        return x