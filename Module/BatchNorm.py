import numpy as np

# @NOTE 全连接层作用在特征维度, 卷积层作用在通道维度
# 公式: y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
# 总体方差: \sigma^2 = \frac{1}{N} * \sum_{i=1}^N(x_i-\x_mu)^2
# 样本方差: \sigma^2 = \frac{1}{N-1} * \sum_{i=1}^N(x_i-\x_mu)^2

# 计算每个batch的方差使用总体方差, 更新running_var使用的是batch的样本方差

# class BatchNorm:
#     def __init__(self,
#                  dims,
#                  num_features,
#                  eps = 1e-5,
#                  momentum =0.1,
#                  ) -> None:
        
#         self.ltype = "BatchNorm"
#         self.dims = dims
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.isTrain = True

#         self.gamma = {
#             "value": np.ones(self.num_features),
#             'grad': np.zeros(self.num_features)
#         }
#         self.beta = {
#             "value": np.zeros(self.num_features),
#             'grad': np.zeros(self.num_features)
#         }
#         # self.running_mean = None
#         # self.running_var = None
#         self.running_mean = np.zeros(self.num_features)
#         self.running_var = np.ones(self.num_features)
#         self.std = None
#         self.mean = None

#         # if self.dims == 2:
#         #     self.axis = 0
#         #     self.re_shape = (-1, self.num_features)
#         # elif self.dims == 4:
#         #     self.axis = (0, 2, 3)
#         #     self.re_shape = (-1, self.num_features, 1, 1)
#     def forward(self, x):
#         if x.ndim == 4:
#             N, C, H, W = x.shape
#             x = x.transpose(0, 2, 3, 1).reshape(-1, C)
#         elif self.dims == 2:
#             N, D = x.shape

#         if self.isTrain:
#             self.mean = np.mean(x, axis=0)
#             self.var = np.var(x, axis=0)
#             sample_var = np.var(x, axis=0, ddof=1)
#             self.std = np.sqrt(self.var + self.eps)
#             self.x_norm = (x - self.mean) / self.std
#             self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*self.mean
#             self.running_var = (1-self.momentum)*self.running_var+self.momentum*sample_var
#         else:
#             self.x_norm = (x-self.running_mean)/self.running_var

#         self.out = self.gamma['value'] * self.x_norm + self.beta['value']
#         if self.dims == 4:
#             self.out = self.out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
#         return self.out
    
    
#     def backward(self, dout):
#         if dout.ndim == 4:
#             N, C, H, W = dout.shape
#             dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
#         elif self.dims == 2:
#             N, D = dout.shape

#         self.gamma['grad'] = np.sum(self.x_norm * dout, axis=0)
#         self.beta['grad'] = np.sum(dout, axis=0)

#         dx_norm = dout * self.gamma['value']
#         dvar = np.sum(dx_norm * (self.x_norm - self.mean), axis=0) * -0.5 * (self.std ** -3)
#         dmean = np.sum(dx_norm * -1 / self.std, axis=0) + dvar * -2 * np.mean(self.x_norm - self.mean, axis=0)
#         dx = dx_norm / self.std + dvar * 2 * (self.x_norm - self.mean) / N + dmean / N
#         if self.dims == 4:
#             dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

#         return dx, self.gamma['grad'], self.beta['grad']


class BatchNorm:
    def __init__(self,
                 dims,
                 num_features,
                 eps = 1e-5,
                 momentum =0.9,
                 ) -> None:
        
        self.ltype = "BatchNorm"
        self.dims = dims
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.isTrain = True

        self.gamma = {
            "value": np.ones(self.num_features),
            'grad': np.zeros(self.num_features)
        }
        self.beta = {
            "value": np.zeros(self.num_features),
            'grad': np.zeros(self.num_features)
        }
        self.running_mean = np.zeros((1, self.num_features))
        self.running_var = np.ones((1, self.num_features))

        # if self.dims == 2:
        #     self.axis = 0
        #     self.re_shape = (-1, self.num_features)
        # elif self.dims == 4:
        #     self.axis = (0, 2, 3)
        #     self.re_shape = (-1, self.num_features, 1, 1)
        

    def forward(self, x):
        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
            self.ndim = 4
        elif x.ndim == 2:
            N, D = x.shape
            self.ndim = 2

        if self.isTrain:
            # print("x shape: ", x.shape)
            x_mean = np.mean(x, axis = 0, keepdims=True)
            x_var = np.var(x, axis = 0, keepdims=True)
            x_sample_var = np.var(x, axis=0, ddof=1, keepdims=True)

            # diff_xmu = x - x_mean
            # sqrtvar = np.sqrt(x_var+self.eps)
            # isqrtvar = 1./sqrtvar
            x_hat = (x - x_mean)/np.sqrt(x_var+self.eps)

            # self.out = self.gamma['value'].reshape(self.re_shape)*x_hat+self.beta['value'].reshape(self.re_shape)
            self.running_mean = self.momentum*self.running_mean+(1-self.momentum)*x_mean
            self.running_var = self.momentum*self.running_var+(1-self.momentum)*x_sample_var
            self.cache = x, x_hat, x_mean, x_var
            # print("x_mean.shape: {}, x_var.shape: {}, self.running_mean.shape: {}, self.running_var.shape: {}".format(x_mean.shape, x_var.shape, self.running_mean.shape, self.running_var.shape))
            # print("x_mean: {}, self.running_mean: {}, x_var: {}, self.running_var: {}".format(x_mean, self.running_mean, x_var, self.running_var))
        else:
            # print(self.running_mean)
            # print(self.running_var)
            x_hat = (x-self.running_mean)/np.sqrt(self.running_var+self.eps)

        self.out = self.gamma['value']*x_hat+self.beta['value']
        if self.ndim == 4:
            self.out = self.out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

        return self.out
       
    

    def backward(self, dout):
        # x, x_hat, x_mean, x_var, diff_xmu, sqrtvar, isqrtvar = self.cache
        x, x_hat, x_mean, x_var = self.cache
        # N = x.shape[0]
        if dout.ndim == 4:
            N,C,H,W = dout.shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
            sz = N*H*W
        else:
            N, D = dout.shape
            sz = N
        # print("dout.shape: ", dout.shape)
        # print("x_hat.shape: ", x_hat.shape)
        self.gamma['grad'] = np.sum(dout*x_hat,axis=0)
        self.beta['grad'] = np.sum(dout, axis=0)

        dxhat = dout * self.gamma['value']
        # divar = np.sum(dxhat*diff_xmu, axis=self.axis)
        # dxmu1 = dxhat * isqrtvar
        # dsqrtvar = -1. /(sqrtvar**2) * divar
        # dvar = 0.5 * 1. /np.sqrt(x_var+self.eps) * dsqrtvar
        # dsq = 1. /N * np.ones(dout.shape) * dvar
        # dxmu2 = 2 * diff_xmu * dsq
        # dx1 = (dxmu1 + dxmu2)
        # dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        # dx2 = 1. /N * np.ones(dout.shape) * dmu
        # dx = dx1 + dx2
        dx = (1./sz) * np.sqrt(x_var+self.eps)* \
             (N*dxhat - np.sum(dxhat, axis=0) - \
              x_hat*np.sum(dxhat*x_hat, axis=0))

        if self.ndim == 4:
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        # dx = (1./N) * np.sqrt(self.running_var+self.eps).reshape(self.re_shape) * \
        #      (N*dxhat - np.sum(dxhat, axis=self.axis).reshape(self.re_shape) - \
        #       x_hat*np.sum(dxhat*x_hat, axis=self.axis).reshape(self.re_shape))

        return dx, self.gamma['grad'], self.beta['grad']



def check_gradient(x, gamma, beta, dx, dgamma, dbeta, dout, param):
    import torch
    '''传入变量为nd.array
    '''
    nF = param['num_features']
    if param['dims'] == 2:
        # x_mean = np.mean(x, axis=0)
        # x_var = np.var(x, axis=0)
        # x_sample_var = np.var(x, axis=0, ddof=1)
        # x_hat = (x-x_mean.reshape(-1, param['num_features']))/np.sqrt(x_var.reshape(-1, param['num_features'])+param['eps'])
        x = torch.tensor(x, requires_grad=True)
        x_mean = torch.mean(x, dim=0)
        x_var = torch.var(x, dim=0)
        # x_mean = torch.tensor(x_mean.reshape(-1, nF))
        # x_var = torch.tensor(x_var.reshape(-1, nF))
        gamma = torch.tensor(gamma.reshape(-1, nF), requires_grad=True)
        beta = torch.tensor(beta.reshape(-1, nF), requires_grad=True)
        x_hat = (x-x_mean)/torch.sqrt(x_var+torch.tensor(param['eps']))
        y = gamma*x_hat+beta
    elif param['dims'] == 4:
        # x_mean = np.mean(x, axis=(0, 2, 3))
        # x_var = np.var(x, axis=(0, 2, 3))
        # x_sample_var = np.var(x, axis=0, ddof=1)
        # x_hat = (x-x_mean.reshape(-1, param['num_features']))/np.sqrt(x_var.reshape(-1, param['num_features'])+param['eps'])
        x = torch.tensor(x, requires_grad=True)
        tmp = torch.mean(x, dim = (0, 2, 3))
        x_mean = torch.mean(x, dim = (0, 2, 3)).reshape(-1, nF, 1, 1)
        x_var = torch.var(x, dim = (0, 2, 3)).reshape(-1, nF, 1, 1)
        # x_mean = torch.tensor(x_mean.reshape(-1, nF, 1, 1))
        # x_var = torch.tensor(x_var.reshape(-1, nF, 1, 1))
        gamma = torch.tensor(gamma.reshape(-1, nF, 1, 1), requires_grad=True)
        beta = torch.tensor(beta.reshape(-1, nF, 1, 1), requires_grad=True)
        x_hat = (x-x_mean)/torch.sqrt(x_var+torch.tensor(param['eps']))
        y = gamma*x_hat+beta
    
    dout = torch.tensor(dout)
    y.backward(dout)
    
    print("diff grad x: ", np.mean(np.abs(dx-x.grad.numpy())))
    print("diff grad gamma: ", np.mean(np.abs(dgamma-gamma.grad.numpy())))
    print("diff grad beta: ", np.mean(np.abs(dbeta-beta.grad.numpy())))


if __name__ == "__main__":
    import torch.nn as nn
    import torch

    dims = 4
    eps = 1e-5
    if dims == 2:
        N, IC = 2, 10
        KH, KW = 3, 3
        stride = 1
        padding = 1
        OC = 3
        image = np.random.normal(0, 1, (N, IC)).astype('float64')
        dout = np.random.rand(N, IC)
        bn = BatchNorm(2, IC, 1e-5)
        bn_pt = nn.BatchNorm1d(IC)
    elif dims == 4:
        N, IC, IH, IW = 2, 5, 4, 4
        KH, KW = 3, 3
        stride = 1
        padding = 1
        OC = IC
        OH = int((IH+2*padding -KH) / stride + 1)
        OW = int((IW+2*padding - KH) / stride + 1)

        image = np.random.normal(0, 1, (N, IC, IH, IW)).astype('float64')
        dout = np.random.rand(N, OC, OH, OW)

        bn = BatchNorm(4, IC, eps)
        bn_pt = nn.BatchNorm2d(IC)
        
    else:
        raise ValueError
    
    y = bn.forward(image)
    x_grad, gamma_grad, beta_grad = bn.backward(dout)
    gamma = bn.gamma['value']
    beta = bn.beta['value']
    param = {
        'num_features': IC,
        'dims': dims,
        'eps': eps
    }
    # check_gradient(image, gamma, beta, x_grad, gamma_grad, beta_grad, dout, param)

    x_tensor = torch.tensor(image, requires_grad=True).to(torch.float32)
    
    y_tensor = bn_pt(x_tensor)
    y_tensor.backward(torch.Tensor(dout))
    
    # print("diff y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    # import torch.nn as nn
    # import torch

    # BatchNorm2d
    N, IC, IH, IW = 2, 5, 4, 4
    KH, KW = 3, 3
    stride = 1
    padding = 1
    OC = IC
    OH = int((IH+2*padding -KH) / stride + 1)
    OW = int((IW+2*padding - KH) / stride + 1)

    image = np.random.normal(0, 1, (N, IC, IH, IW)).astype('float64')
    dout = np.random.rand(N, OC, OH, OW)
    # print(image)

    ## numpy实现的卷积输出
    bn = BatchNorm(4, IC, 1e-5)
    y = bn.forward(image)
    x_grad, gamma_grad, beta_grad = bn.backward(dout)
    # _, w_grad, b_grad = conv.backward(dout)

    # pytorch卷积输出
    x_tensor = torch.tensor(image).to(torch.float32)
    bn_pt = nn.BatchNorm2d(IC)
    y_tensor = bn_pt(x_tensor)
    y_tensor.backward(torch.Tensor(dout))
    
    print("diff y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    print("diff running_mean: ", np.mean(np.abs(bn.running_mean - bn_pt.running_mean.detach().numpy())))
    print("diff running_var: ", np.mean(np.abs(bn.running_var - bn_pt.running_var.detach().numpy())))
    # print("diff grad x: ", np.mean(np.abs(x_grad-x_tensor.grad.numpy())))
    print("diff grad w: ", np.mean(np.abs(gamma_grad-bn_pt.weight.grad.numpy())))
    print("diff grad b: ", np.mean(np.abs(beta_grad-bn_pt.bias.grad.numpy())))


    # # BatchNorm1d
    # N, IC = 2, 10
    # KH, KW = 3, 3
    # stride = 1
    # padding = 1
    # OC = 3
    

    # image = np.random.normal(0, 1, (N, IC)).astype('float64')
    # dout = np.random.rand(N, IC)
    # # print(image)

    # ## numpy实现的卷积输出
    # bn = BatchNorm(2, IC, 1e-5)
    # y = bn.forward(image)
    # _, gamma_grad, beta_grad = bn.backward(dout)

    # # pytorch卷积输出
    # x_tensor = torch.Tensor(image)
    # bn_pt = nn.BatchNorm1d(IC)
    # y_tensor = bn_pt(x_tensor)
    # y_tensor.backward(torch.Tensor(dout))
    
    # print("diff y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    # print("diff running_mean: ", np.mean(np.abs(bn.running_mean - bn_pt.running_mean.detach().numpy())))
    # print("diff running_var: ", np.mean(np.abs(bn.running_var - bn_pt.running_var.detach().numpy())))
    # print("diff grad w: ", np.mean(np.abs(gamma_grad-bn_pt.weight.grad.numpy())))
    # print("diff grad b: ", np.mean(np.abs(beta_grad-bn_pt.bias.grad.numpy())))