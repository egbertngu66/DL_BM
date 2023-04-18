import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import BMDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import config as cfg
from test import test
from model import Net
from utils import torch_utils


def train(model,
          data_config,
          ntype,
          batch_size = 32,
          epochs = 10000,
          resume = False,   # 是否重新训练
          weights_path = "weights"):
    device = torch_utils.select_device()
    model.to(device)
    weights_path = os.path.join(weights_path, ntype)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    
    latest_weights_file = os.path.join(weights_path, 'latest.pt')
    best_weights_file = os.path.join(weights_path, 'best.pt')
    # batch_size = config["batch_size"]
    # epochs = config["epochs"]
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 数据集的config解析
    data_fpath = data_config["data_path"]
    if "data_min" in data_config and "data_max" in data_config:
        data_min = np.load(data_config["data_min"], allow_pickle= True)
        data_max = np.load(data_config["data_max"], allow_pickle= True)
    else:
        raise ValueError("Please run script to get data_min.npy and data_max.npy")
    if "data_mu" in data_config and "data_sigma" in data_config:
        data_mu = np.load(data_config["data_mu"], allow_pickle= True)
        data_sigma = np.load(data_config["data_sigma"], allow_pickle= True)
    else:
        raise ValueError("Please run script to get data_mu.npy and data_sigma.npy")
    
    train_dataset = BMDataset(data_fpath, data_min, data_max, data_mu, data_sigma, ntype, "train")
    train_loader = DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=batch_size)
    test_dataset = BMDataset(data_fpath, data_min, data_max, data_mu, data_sigma, ntype, "valid")
    test_loader = DataLoader(test_dataset,
                        shuffle=True,
                        batch_size=batch_size)
    
    lr0 = 0.001
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=0.9)
    # scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)
    if resume:
        start_epoch = 0
        best_loss = float("inf")
    else:
        checkpoint = torch.load(latest_weights_file)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_loss = checkpoint['best_loss']
        del checkpoint
        
    model.train()
    train_loss = list()
    test_loss = list()
    for epoch in range(start_epoch, epochs, 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        train_loss = test(model, train_loader, criterion)
        test_loss = test(model, test_loader, criterion)

        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, latest_weights_file)

        if test_loss < best_loss:
            print('===epoch: {} === Test set: Average loss: {:.8f}'.format(epoch+1, test_loss))
            best_loss = test_loss
            os.system('cp {} {}'.format(
                latest_weights_file,
                best_weights_file,
            ))
    # # min_test_loss = np.inf
    # # for epoch in range(epochs):
    # #     model.train()
    # #     for batch_idx, (data, target) in enumerate(train_loader):
    # #         data, target = data.to(device), target.to(device)
    # #         optimizer.zero_grad()
    # #         output = model(data)
    # #         loss = criterion(output, target)
    # #         loss.backward()
    # #         optimizer.step()
    # #         scheduler.step()
    # #     test_loss = test(model, test_loader, criterion, device)
    #     # print(test_loss)
    #     # 保存模型
    #     if test_loss<min_test_loss:
    #         print('===epoch: {} === Test set: Average loss: {:.4f}'.format(epoch+1, test_loss))
    #         checkpoint = {
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'epoch': epoch
    #         }
    #         torch.save(checkpoint, 'checkpoints/checkpoint.pth')
    #     min_test_loss = min_test_loss if min_test_loss<test_loss else test_loss

if __name__ == "__main__":
    ntype = cfg.config["ntype"]
    model = Net(ntype)
    # if os.path.isfile(cfg.config["ckpt"]):
    #     checkpoint = torch.load(cfg.config["ckpt"])
    #     # print(checkpoint)
    #     model.load_state_dict(checkpoint['model'])
    batch_size = cfg.config["batch_size"]
    epochs = cfg.config["epochs"]
    train(model, cfg.config, ntype, batch_size, epochs, True)