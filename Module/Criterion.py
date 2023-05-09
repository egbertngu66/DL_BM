import numpy as np


class MSELoss:

    def __init__(self, reduction = 'sum') -> None:
        self.reduction = reduction
        self.ltype = "Criterion"
        self.cache = None

    def forward(self, target, pred):
        if self.reduction == 'mean':
            self.loss = np.mean(np.square(target-pred))
        elif self.reduction == 'sum':
            self.loss = np.sum(np.square(target-pred))
        self.cache = target, pred

        return self.loss
    
    def backward(self):
        target, pred = self.cache
        # print("target.size:", target.size)
        if self.reduction == 'mean':
            dout = 2*(pred-target) / target.size
        elif self.reduction == 'sum':
            dout = 2*(pred-target)

        return dout
    

def check_gradient(y_target, y_pred, dy):
    import torch
    '''传入变量为nd.array
    '''
    y_pred = torch.tensor(y_pred, requires_grad=True)
    # w = torch.tensor(w)
    y_target = torch.tensor(y_target, requires_grad=False)
    
    loss = torch.mean(torch.square(y_target-y_pred))
    loss.backward(torch.ones(y_target.size))
    
    print("diff grad x: ", np.mean(np.abs(dy-y_pred.grad.numpy())))


if __name__ == "__main__":

    N = 10
    F  = 64
    