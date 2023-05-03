# # !/usr/bin/python
# # -*- coding: UTF-8 -*-
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import Module
# import numpy as np


# def weight_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform(m.weight.data)
#     elif isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform(m.weight.data)
#         m.bias.data.fill_(1)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight.data, 1)
#         nn.init.constant_(m.bias.data, 0)                                                                                                                                                                                                                                                                                                                                                                                                                                                           


# class Net(Module.Model):
#     def __init__(self, ntype, lr, inshape):
#         super(Net, self).__init__(lr)
#         self.ntype = ntype
#         if self.ntype == "Linear":
#             self.backbone = Module.Linear(9, 128)
#             self.bn = Module.BatchNorm(2, 128)
#         elif self.ntype == "Conv":
#             self.backbone = Module.Conv2d(inshape, 128, ksize=3, stride=1, padding=0)
#             self.bn = Module.BatchNorm(4, 128)
#         else:
#             raise ValueError("self.ntype can only take \"Linear\" and \"Conv\"")

#         self.relu = Module.Relu()

#         self.fc1 = Module.Linear(128, 64)
#         self.bn1 = Module.BatchNorm(2, 64)
#         self.relu1 = Module.Relu()

#         self.fc2 = Module.Linear(64, 64)
#         self. bn2 = Module.BatchNorm(2, 64)
#         self.relu2 = Module.Relu()

#         self.fc3 = Module.Linear(64, 1)
#         self.sigmoid = Module.Sigmoid()
        
#         self.layers = [
#             self.backbone,
#             self.bn,
#             self.relu,
#             self.fc1,
#             self.bn1,
#             self.relu1,
#             self.fc2,
#             self.bn2,
#             self.relu2,
#             self.fc3,
#             self.sigmoid
#         ]

        
#     def forward(self, x):
#         # 3*3*1 --> 1*1*128
#         x = self.backbone.forward(x)
#         x = self.bn.forward(x)
#         x = self.relu.forward(x)

#         x = x.reshape(-1, 128)
#         # 1*128 --> 1*64
#         x = self.fc1.forward(x)
#         x = self.bn1.forward(x)
#         x = self.relu1.forward(x)
#         # 1*64 --> 1*64
#         x = self.fc2.forward(x)
#         x = self.bn2.forward(x)
#         x = self.relu2.forward(x)
#         # 1*64 --> 1*1
#         x = self.fc3.forward(x)
#         x = self.sigmoid.forward(x)
#         x = x.reshape(-1)
        
#         return x

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
    def __init__(self, ntype, lr, inshape):
        super(Net, self).__init__(lr)
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
        
        self.layers = [
            self.backbone,
            # self.bn,
            self.relu,
            self.fc1,
            # self.bn1,
            self.relu1,
            self.fc2,
            # self.bn2,
            self.relu2,
            self.fc3,
            self.sigmoid
        ]

        
    def forward(self, x):
        # 3*3*1 --> 1*1*128
        x = self.backbone.forward(x)
        # x = self.bn.forward(x)
        x = self.relu.forward(x)

        x = x.reshape(-1, 128)
        # 1*128 --> 1*64
        x = self.fc1.forward(x)
        # x = self.bn1.forward(x)
        x = self.relu1.forward(x)
        # 1*64 --> 1*64
        x = self.fc2.forward(x)
        # x = self.bn2.forward(x)
        x = self.relu2.forward(x)
        # 1*64 --> 1*1
        x = self.fc3.forward(x)
        x = self.sigmoid.forward(x)
        x = x.reshape(-1)
        
        return x