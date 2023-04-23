import pandas as pd
import numpy as np
import os
ROOT_PATH = os.path.abspath(__file__).split('/utils/')[0]
import sys
sys.path.append(ROOT_PATH)
import config as cfg


def cal_min_max(data_path, save_path):
    '''计算数据的mu和sigma
    Args: 
        fpath: excel的路径, str
    Returns:
        mu: 均值
        sigma: 方差
    '''

    sheet_name = "Literature data"

    df=pd.read_excel(data_path, sheet_name=sheet_name)
    data = df.values
    data = data[:, 1:-3]
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    np.save(save_path['min'], min)
    np.save(save_path['max'], max)
    return min, max

if __name__ == "__main__":
    data_path = os.path.join(ROOT_PATH, cfg.config['data_path'])
    save_path = {
        'min': os.path.join(ROOT_PATH, cfg.config['data_min']),
        'max': os.path.join(ROOT_PATH, cfg.config['data_max'])
    }
    cal_min_max(data_path, save_path)