import torch
import numpy as np
from model import Net
from dataset import BMDataset
from config import config
from torch.utils.data import DataLoader

# def test(model, test_loader, criterion):
#     device = next(model.parameters()).device
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             # print(output, target, criterion(output, target).item())
#     test_loss /= len(test_loader.dataset)
#     # print(len(test_loader.dataset))

#     return test_loss

def test(model, test_loader, criterion):
    # device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    for data, target in test_loader:
        data, target = data.numpy(), target.numpy()
        output = model.forward(data)
        test_loss += criterion.forward(output, target)
        # output_dlabel = test_loader.dataset.decode_label(output)
        # target_dlabel = test_loader.dataset.decode_label(target)
        # print("predicts:\n{}\nlabels:\n{}".format(output_dlabel, target_dlabel))
        # print(output, target, criterion(output, target).item())
    test_loss /= len(test_loader.dataset)
    # print(test_loss)
    print(len(test_loader.dataset))

    return test_loss