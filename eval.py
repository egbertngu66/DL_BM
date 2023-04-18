from utils import torch_utils
import torch
import numpy as np
import config as cfg
from model import Net
from dataset import BMDataset
from utils.eval_indicators import Indicators


def inference(model, data):
    device = torch_utils.select_device()
    model.to(device)
    model.eval()
    with torch.no_grad():
        if model.ntype == "Linear":
            data = data.view(1, -1)
        elif model.ntype == "Conv":
            data = data.view(1, 1, 3, 3)
        data = data.to(device)
        output = model(data)
    return output


def get_predicts_labels(model, dataset):
    '''获取数据集的预测值和真实值
    Args:
        model: 训练后的模型
        dataset: BMDataset
    Returns:
        predicts: 预测值, shape = (N, ), np.array
        labels: 样本标签值, shape = (N, ), np.array
    '''
    
    predicts = list()
    labels = list()
    
    for data, target in dataset:
        predict = inference(model, data)
        predict = dataset.decode_label(predict.cpu().item())
        target = dataset.decode_label(target.item())
        predicts.append(predict)
        labels.append(target)
        # predicts.append(predict.cpu().item())
        # labels.append(target.item())
    predicts = np.array(predicts)
    labels = np.array(labels)
    print("predicts:\n{}\nlabels:\n{}".format(predicts, labels))

    return predicts, labels


def main():
    ntype = cfg.config["ntype"]
    model = Net(ntype)
    checkpoint = torch.load(cfg.config["ckpt"])
    # print(checkpoint)
    model.load_state_dict(checkpoint['model'])

    data_fpath = cfg.config["data_path"]
    if "data_min" in cfg.config and "data_max" in cfg.config:
        data_min = np.load(cfg.config["data_min"], allow_pickle= True)
        data_max = np.load(cfg.config["data_max"], allow_pickle= True)
    else:
        raise ValueError("Please run script to get data_min.npy and data_max.npy")
    if "data_mu" in cfg.config and "data_sigma" in cfg.config:
        data_mu = np.load(cfg.config["data_mu"], allow_pickle= True)
        data_sigma = np.load(cfg.config["data_sigma"], allow_pickle= True)
    else:
        raise ValueError("Please run script to get data_mu.npy and data_sigma.npy")
    train_dataset = BMDataset(data_fpath, data_min, data_max, data_mu, data_sigma, ntype, "train")
    valid_dataset = BMDataset(data_fpath, data_min, data_max, data_mu, data_sigma, ntype, "valid")
    test_dataset = BMDataset(data_fpath, data_min, data_max, data_mu, data_sigma, ntype, "test")

    train_preds, train_labels = get_predicts_labels(model, train_dataset)
    train_indicators = Indicators(train_preds, train_labels)
    train_R2 = train_indicators.R_square()
    train_MAPE = train_indicators.MAPE()
    train_RMSE = train_indicators.RMSE()
    train_MAE = train_indicators.MAE()
    print("***Training set*** R2: {}, MAPE: {}, RMSE: {}, MAE: {}".format(train_R2, train_MAPE, train_RMSE, train_MAE))

    valid_preds, valid_labels = get_predicts_labels(model, valid_dataset)
    valid_indicators = Indicators(valid_preds, valid_labels)
    valid_R2 = valid_indicators.R_square()
    valid_MAPE = valid_indicators.MAPE()
    valid_RMSE = valid_indicators.RMSE()
    valid_MAE = valid_indicators.MAE()
    print("***Valid set*** R2: {}, MAPE: {}, RMSE: {}, MAE: {}".format(valid_R2, valid_MAPE, valid_RMSE, valid_MAE))

    test_preds, test_labels = get_predicts_labels(model, test_dataset)
    test_indicators = Indicators(test_preds, test_labels)
    test_R2 = test_indicators.R_square()
    test_MAPE = test_indicators.MAPE()
    test_RMSE = test_indicators.RMSE()
    test_MAE = test_indicators.MAE()
    print("***Test set*** R2: {}, MAPE: {}, RMSE: {}, MAE: {}".format(test_R2, test_MAPE, test_RMSE, test_MAE))


if __name__ == "__main__":
    main()