# !/usr/bin/python
# -*- coding: UTF-8 -*-
import torch


CUDA_AVAILABLE = torch.cuda.is_available()


def init_seeds(seed=0):
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def select_device(force_cpu=False):
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')
    return device
