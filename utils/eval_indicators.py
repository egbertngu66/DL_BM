import numpy as np


class Indicators:
    def __init__(self, predicts, labels) -> None:
        assert(predicts.shape == labels.shape)
        self.len = predicts.shape[0]
        self.predicts = predicts
        self.labels = labels

    def R_square(self):
        ave_label = np.mean(self.labels)
        diff_p_a = 0.0
        diff_a_avea = 0.0

        for i in range(self.len):
            diff_p_a += np.square(self.predicts[i] - self.labels[i])
            diff_a_avea += np.square(self.labels[i]-ave_label)

        R2 = 1-diff_p_a/diff_a_avea
        return R2

    def MAPE(self):
        mape = 0.0
        for i in range(self.len):
            mape += np.abs((self.labels[i]-self.predicts[i])/self.labels[i])
        
        mape /= self.len
        return mape


    def RMSE(self):
        rmse = 0.0
        for i in range(self.len):
            rmse += np.square(self.labels[i]-self.predicts[i])
        
        rmse = np.sqrt(rmse/self.len)
        return rmse

    def MAE(self):
        mae = 0.0
        for i in range(self.len):
            mae += np.abs(self.labels[i]-self.predicts[i])
        
        mae /= self.len
        return mae
