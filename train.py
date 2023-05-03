import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import BMDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import config as cfg
from test import test
from model import Net, weight_init
from utils import torch_utils, display
import Module


def save_model_param_as_excel(save_path, model_param):
    '''将模型的权重文件保存为excel
    Args: 
        model_param: 模型参数, dict
    Returns:
        None
    '''
    import pandas as pd
    writer = pd.ExcelWriter(save_path, engine='openpyxl')

    for key, value in model_param.items():
        if 'bn' in key:
            continue
        # print("key\n:{}\nvalue:\n:{}".format(key, value.size()))
        value = value.cpu().numpy()
        if value.ndim == 4:
            OC, IC, KH, KW = value.shape
            value_new = np.ones((KH*OC+OC-1, KW*IC+IC-1)) * np.NAN

            for i in range(OC):
                for j in range(IC):
                    value_new[i*KH+i:(i+1)*KH+i, j*KW+j:(j+1)*KW+j] = value[i, j, :, :]
            value = value_new
        value_frame = pd.DataFrame(value)
        value_frame.to_excel(excel_writer=writer, sheet_name=key, header=None, index=False)
    
    writer.close()

def train(model,
          data_config,
          ntype,
          lr0 = 0.01,
          batch_size = 32,
          epochs = 2000,
          resume = False,   # 是否重新训练
          weights_path = "weights/Paper"):
    # device = torch_utils.select_device()
    # model.to(device)
    weights_path = os.path.join(weights_path, ntype)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    
    latest_weights_file = os.path.join(weights_path, 'latest.pt')
    best_weights_file = os.path.join(weights_path, 'best.pt')
    # batch_size = config["batch_size"]
    # epochs = config["epochs"]
    criterion = Module.MSELoss()
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
                              batch_size=batch_size,
                              drop_last=True)
    test_dataset = BMDataset(data_fpath, data_min, data_max, data_mu, data_sigma, ntype, "valid")
    test_loader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=batch_size,
                             drop_last=True)
    
    start_epoch = 0
    for epoch in range(start_epoch, epochs, 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            print("epoch: {}, batch_idx: {}".format(epoch, batch_idx))
            data, target = data.numpy(), target.numpy()
            output = model.forward(data)
            loss = criterion.forward(output, target)
            dout = criterion.backward()
            model.backward(dout)
            print("Training loss: {}".format(loss))

    # # lr0 = 0.001
    # # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, momentum=0.9)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr0, betas=(0.9, 0.999), eps=1e-08)
    # # step_size 是按batch迭代次数算
    # scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

    # if resume:
    #     start_epoch = 0
    #     best_loss = float("inf")
    #     # model.apply(weight_init)
    # else:
    #     checkpoint = torch.load(latest_weights_file)
    #     model.load_state_dict(checkpoint['model'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     if checkpoint['optimizer'] is not None:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         best_loss = checkpoint['best_loss']
    #     del checkpoint
        
    # model.train()
    # train_loss_list = list()
    # test_loss_list = list()
    # for epoch in range(start_epoch, epochs, 1):
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         # print("lr = {}".format(optimizer.param_groups[0]["lr"]))
    #         scheduler.step()
    #     train_loss = test(model, train_loader, criterion)
    #     test_loss = test(model, test_loader, criterion)

    #     # if (epoch+1)%5 == 0:
    #     train_loss_list.append(train_loss)
    #     test_loss_list.append(test_loss)

    #     checkpoint = {'epoch': epoch,
    #                   'best_loss': best_loss,
    #                   'model': model.state_dict(),
    #                   'optimizer': optimizer.state_dict()}
    #     torch.save(checkpoint, latest_weights_file)

    #     if test_loss < best_loss:
    #         print('===epoch: {} === Test set: Average loss: {:.8f}'.format(epoch+1, test_loss))
    #         best_loss = test_loss
    #         # os.system('cp {} {}'.format(
    #         #     latest_weights_file,
    #         #     best_weights_file,
    #         # ))
    #         shutil.copy(latest_weights_file, best_weights_file)
        
    #     # 将模型参数保存到excel
    #     checkpoint_best = torch.load(best_weights_file)
    #     excel_save_path = best_weights_file.replace('pt', "xlsx")
    #     save_model_param_as_excel(excel_save_path, checkpoint_best['model'])
            
    # display.draw_loss(train_loss_list, test_loss_list)



if __name__ == "__main__":
    ntype = cfg.config["ntype"]
    lr0 = cfg.config["lr0"]
    weights_path = cfg.config["weights_path"]
    batch_size = cfg.config["batch_size"]
    epochs = cfg.config["epochs"]
    if ntype == "Conv":
        inshape = (batch_size, 1, 3, 3)
    elif ntype == "Linear":
        inshape = (batch_size, 9)
    else:
        pass

    model = Net(ntype, lr0, inshape)
    # if os.path.isfile(cfg.config["ckpt"]):
    #     checkpoint = torch.load(cfg.config["ckpt"])
    #     # print(checkpoint)
    #     model.load_state_dict(checkpoint['model'])
    
    train(model, cfg.config, ntype, lr0=lr0, batch_size=batch_size, epochs=epochs, resume=True, weights_path=weights_path)

    # ntype = cfg.config["ntype"]
    # model = Net(ntype)
    # # if os.path.isfile(cfg.config["ckpt"]):
    # #     checkpoint = torch.load(cfg.config["ckpt"])
    # #     # print(checkpoint)
    # #     model.load_state_dict(checkpoint['model'])
    # lr0 = cfg.config["lr0"]
    # weights_path = cfg.config["weights_path"]
    # batch_size = cfg.config["batch_size"]
    # epochs = cfg.config["epochs"]
    # train(model, cfg.config, ntype, lr0=lr0, batch_size=batch_size, epochs=epochs, resume=True, weights_path=weights_path)