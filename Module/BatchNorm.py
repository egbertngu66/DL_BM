import numpy as np

# @NOTE 全连接层作用在特征维度, 卷积层作用在通道维度
# 公式: y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
# 总体方差: \sigma^2 = \frac{1}{N} * \sum_{i=1}^N(x_i-\x_mu)^2
# 样本方差: \sigma^2 = \frac{1}{N-1} * \sum_{i=1}^N(x_i-\x_mu)^2

# 计算每个batch的方差使用总体方差, 更新running_var使用的是batch的样本方差
class BatchNorm:
    def __init__(self,
                 dims,
                 num_features,
                 eps = 1e-5,
                 momentum =0.1,
                 ) -> None:
        
        self.ltype = "BatchNorm"
        self.dims = dims
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # self.affine = affine
        # self.track_running_stats = track_running_stats

        self.gamma = {
            "value": np.ones(self.num_features),
            'grad': np.zeros(self.num_features)
        }
        self.beta = {
            "value": np.zeros(self.num_features),
            'grad': np.zeros(self.num_features)
        }
        self.running_mean = np.zeros(self.num_features)
        self.running_var = np.ones(self.num_features)

    def forward(self, x):
        
        if self.dims == 2:
            x_mean = np.mean(x, axis=0)
            x_var = np.var(x, axis=0)
            x_sample_var = np.var(x, axis=0, ddof=1)
            x_hat = (x-x_mean.reshape(-1, self.num_features))/np.sqrt(x_var.reshape(-1, self.num_features)+self.eps)
            self.out = self.gamma['value'].reshape(-1, self.num_features)*x_hat+self.beta['value'].reshape(-1, self.num_features)
        elif self.dims == 4:
            x_mean = np.mean(x, axis=(0, 2, 3))
            x_var = np.var(x, axis=(0, 2, 3))
            x_sample_var = np.var(x, axis=(0, 2, 3), ddof=1)
            # print(x_var, np.mean(np.square(x-x_mean.reshape(-1, self.num_features, 1, 1)),axis = (0, 2, 3)))
            x_hat = (x-x_mean.reshape(-1, self.num_features, 1, 1))/np.sqrt(x_var.reshape(-1, self.num_features, 1, 1)+self.eps)
            self.out = self.gamma['value'].reshape(-1, self.num_features, 1, 1)*x_hat+self.beta['value'].reshape(-1, self.num_features, 1, 1)
        self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*x_mean
        self.running_var = (1-self.momentum)*self.running_var+self.momentum*x_sample_var
        self.cache = x, x_hat

        return self.out
    

    def backward(self, dout):
        x, x_hat = self.cache
        N = x.shape[0]
        if self.dims == 2:
            axis = 0
            re_shape = (-1, self.num_features)
        elif self.dims == 4:
            axis = (0, 2, 3)
            re_shape = (-1, self.num_features, 1, 1)
        # print(dout*x_hat)
        self.gamma['grad'] = np.sum(dout*x_hat,axis=axis)
        self.beta['grad'] = np.sum(dout, axis=axis)

        d_xhat  = dout * self.gamma['value'].reshape(re_shape)

        dx = (1./N) * np.sqrt(self.running_var+self.eps).reshape(re_shape) * \
             (N*d_xhat - np.sum(d_xhat, axis=axis).reshape(re_shape) - \
              x_hat*np.sum(d_xhat*x_hat, axis=axis).reshape(re_shape))

        return dx, self.gamma['grad'], self.beta['grad']


def check_gradient(x, gamma, beta, dx, dgamma, dbeta, dout, param):
    import torch
    '''传入变量为nd.array
    '''
    nF = param['num_features']
    if param['dims'] == 2:
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        # x_sample_var = np.var(x, axis=0, ddof=1)
        # x_hat = (x-x_mean.reshape(-1, param['num_features']))/np.sqrt(x_var.reshape(-1, param['num_features'])+param['eps'])
        x = torch.tensor(x, requires_grad=True)
        x_mean = torch.tensor(x_mean.reshape(-1, nF))
        x_var = torch.tensor(x_var.reshape(-1, nF))
        gamma = torch.tensor(gamma.reshape(-1, nF), requires_grad=True)
        beta = torch.tensor(beta.reshape(-1, nF), requires_grad=True)
        x_hat = (x-x_mean)/torch.sqrt(x_var+torch.tensor(param['eps']))
        y = gamma*x_hat+beta
    elif param['dims'] == 4:
        x_mean = np.mean(x, axis=(0, 2, 3))
        x_var = np.var(x, axis=(0, 2, 3))
        # x_sample_var = np.var(x, axis=0, ddof=1)
        # x_hat = (x-x_mean.reshape(-1, param['num_features']))/np.sqrt(x_var.reshape(-1, param['num_features'])+param['eps'])
        x = torch.tensor(x, requires_grad=True)
        x_mean = torch.tensor(x_mean.reshape(-1, nF, 1, 1))
        x_var = torch.tensor(x_var.reshape(-1, nF, 1, 1))
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
    # dims = 2
    # eps = 1e-5
    # if dims == 2:
    #     N, IC = 2, 10
    #     KH, KW = 3, 3
    #     stride = 1
    #     padding = 1
    #     OC = 3
    #     image = np.random.normal(0, 1, (N, IC)).astype('float64')
    #     dout = np.random.rand(N, IC)
    #     bn = BatchNorm(2, IC, 1e-5)
    # elif dims == 4:
    #     N, IC, IH, IW = 2, 2, 4, 4
    #     KH, KW = 3, 3
    #     stride = 1
    #     padding = 1
    #     OC = IC
    #     OH = int((IH+2*padding -KH) / stride + 1)
    #     OW = int((IW+2*padding - KH) / stride + 1)

    #     image = np.random.normal(0, 1, (N, IC, IH, IW)).astype('float64')
    #     dout = np.random.rand(N, OC, OH, OW)

    #     bn = BatchNorm(4, IC, eps)
        
    # else:
    #     raise ValueError
    
    # y = bn.forward(image)
    # x_grad, gamma_grad, beta_grad = bn.backward(dout)
    # gamma = bn.gamma['value']
    # beta = bn.beta['value']
    # param = {
    #     'num_features': IC,
    #     'dims': dims,
    #     'eps': eps
    # }
    # check_gradient(image, gamma, beta, x_grad, gamma_grad, beta_grad, dout, param)
    import torch.nn as nn
    import torch

    # BatchNorm2d
    N, IC, IH, IW = 2, 2, 4, 4
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
    x_tensor = torch.tensor(image, requires_grad=True).to(torch.float32)
    bn_pt = nn.BatchNorm2d(IC)
    y_tensor = bn_pt(x_tensor)
    y_tensor.backward(torch.Tensor(dout))
    
    print("diff y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    print("diff running_mean: ", np.mean(np.abs(bn.running_mean - bn_pt.running_mean.detach().numpy())))
    print("diff running_var: ", np.mean(np.abs(bn.running_var - bn_pt.running_var.detach().numpy())))
    print(x_tensor.grad)
    print("diff grad x: ", np.mean(np.abs(x_grad-x_tensor.grad.numpy())))
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