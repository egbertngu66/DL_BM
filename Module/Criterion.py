import numpy as np


class MSELoss:

    def __init__(self) -> None:
        self.ltype = "Criterion"
        self.cache = None

    def forward(self, target, pred):
        self.loss = np.mean(np.square(target-pred))
        self.cache = target, pred

        return self.loss
    
    def backward(self):
        target, pred = self.cache
        dout = -2*(pred-target) / target.size

        return dout