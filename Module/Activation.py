import numpy as np


class Sigmoid:

    def __init__(self):
        self.ltype = "Activation"
        self.cache = None

    def forward(self, x):
        self.out = 1/(1 + np.exp(-x))
        self.cache = self.out

        return self.out
    
    def backward(self, dout):
        out = self.cache
        x_grad = dout * out * (1 - out)

        return x_grad


class Relu:

    def __init__(self):
        self.ltype = "Activation"
        self.cache = None

    
    def forward(self, x):
        self.out = np.maximum(0, x)
        self.cache = self.out

        return self.out
    
    
    def backward(self, dout):
        x = self.cache
        x_grad = dout
        x_grad[x<=0] = 0

        return x_grad


def check_gradient(x, fc, dx, dout, param = None):
    import torch
    '''传入变量为nd.array
    '''
    x = torch.tensor(x, requires_grad=True)
    dout = torch.tensor(dout, requires_grad=True)
    y = fc(x)
    
    y.backward(dout)
    print("diff grad x: ", np.mean(np.abs(dx-x.grad.numpy())))


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    def sigmoid(x):
        return 1/(1 + torch.exp(-x))

    dims = 4
    if dims  == 4:
        N, IC, IH, IW = 1, 1, 4, 4
        image = np.random.normal(0, 1, (N, IC, IH, IW)).astype('float64')
        dout = np.random.rand(N, IC, IH, IW)

    ## 验证sigmoid
    sigmoidc = Sigmoid()
    y = sigmoidc.forward(image)
    x_grad = sigmoidc.backward(dout)

    x_tensor = torch.Tensor(image).double()
    y_tensor = torch.sigmoid(x_tensor)
    print("diff grad y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    check_gradient(image, sigmoid, x_grad, dout, param = None)

    ## 验证relu
    reluc = Relu()
    y = reluc.forward(image)
    x_grad = reluc.backward(dout)

    x_tensor = torch.tensor(image, requires_grad=True).double()
    y_tensor = F.relu(x_tensor)
    y_tensor.backward(torch.Tensor(dout))
    print("diff grad y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    print("diff grad x: ", np.mean(np.abs(x_grad-x_tensor.grad.numpy())))