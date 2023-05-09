import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


def  load_data(fpath, dtype):
    '''从excel中加载数据, 只加载data和label对应的十列数据, 标签顺序如下:
    [cement, water, sand, natrual aggreate, recycled aggrate, fly ash, silica fume, slag, slump, 28d compressive strength]

    Args: 
        fpath: excel的路径, str
        dtype: [train | test | valid]. train训练集, valid训练的验证集, test模型的测试集
    Returns:
        data: 仅包含数据和标签, ndarray, shape=(N, 10)
    '''
    
    # sheet_name: ["Literature data" | "Experimental data"], str
    if dtype == "train" or dtype == "valid":
        sheet_name = "Literature data"
    elif dtype == "test":
        sheet_name = "Experimental data"
    else:
        raise ValueError("dtype can only take \"train\", \"valid\" and \"test\"")
    df=pd.read_excel(fpath, sheet_name=sheet_name)
    data = df.values

    if dtype == "train":
        data = data[:, 1:-1]
        data = data[np.where(data[:, -1] == "Training set")]
        data = data[:, :-2]
    elif dtype == "valid":
        data = data[:, 1:-1]
        data = data[np.where(data[:, -1] == "Test  set")]
        data = data[:, :-2]
    elif dtype == "test":
        data = data[:, 1:-1]
    else:
        pass

    return data


class BMDataset(Dataset):
    # @TODO 模型验证，未带标签的数据加载
    def __init__(self, 
                 data_fpath,
                 data_min,
                 data_max, 
                 data_mu,
                 data_sigma,
                 ntype,     # 网络类型 ["Linear | "Conv"]
                 dtype,     # 数据集类型 ["train" | "valid" | "test"]
                 transform = None) -> None:
        
        self.min = data_min
        self.max = data_max
        self.mu = data_mu
        self.sigma = data_sigma

        # 加载数据
        data_all = load_data(data_fpath, dtype).astype(np.float32)
        # print(data_all)
        # self.label = data_all[:, -1].astype(np.float32)
        self.data_all = (data_all - self.min)/(self.max - self.min)
        # print(data_all)
        # self.data_all = (self.data_all-self.mu)/self.sigma
        self.data = self.data_all[:, :-1].astype(np.float32)
        self.label = self.data_all[:, -1].astype(np.float32)

        self.ntype = ntype
        self.transform = transform
        
    def __len__(self):
        return self.data_all.shape[0]
    
    def __getitem__(self, index):
        # @TODO 变量的排列顺序未考虑
        data = self.data[index]
        if self.ntype == "Linear":
            data = torch.tensor(data)
        elif self.ntype == "Conv":
            data = torch.tensor(data.reshape(1, 3, 3))
        else:
            raise ValueError("self.ntype can only take \"Linear\" and \"Conv\"")
        
        label = torch.tensor(self.label[index])

        return data, label
    
    
    def decode_label(self, label):
        # dlabel = label*self.sigma[-1] + self.mu[-1]
        dlabel = label * (self.max[-1]-self.min[-1]) + self.min[-1]

        return dlabel


if __name__ == "__main__":
    from config import config
    data_fpath = config["data_path"]
    # 加载数据集变量的最大最小值
    if "data_min" in config and "data_max" in config:
        data_min = np.load(config["data_min"], allow_pickle= True)
        data_max = np.load(config["data_max"], allow_pickle= True)
    else:
        raise ValueError("Please run script utils/cal_min_max.py to get data_min.npy and data_max.npy")
    # 加载数据集变量的均值和方差
    if "data_mu" in config and "data_sigma" in config:
        data_mu = np.load(config["data_mu"], allow_pickle= True)
        data_sigma = np.load(config["data_sigma"], allow_pickle= True)
    else:
        raise ValueError("Please run script to get data_mu.npy and data_sigma.npy")
    ntype = config["ntype"]
    bm_dataset = BMDataset(data_fpath, data_min, data_max, data_mu, data_sigma, ntype, "train")
    print(bm_dataset.__getitem__(5))