import numpy as np
# @TODO stride不为1的时候存在bug


def linear(x, w, b=None):
    '''全连接层的前向计算.
    Args:
        x: 全连接层的输入, nd.array, shape = (N, IF)
        w: 全连接层的权重, nd.array, shape = (OF, IF)
        b: 全连接层的偏置, nd.array, shape = (OF, )
    Returns:
        self.out: 全连接层的输出, nd.array, shape = (N, OF)
    '''
    out = np.dot(x, w.T)
    if b is not None:
        out = out + b
    
    return out


class Linear(object):
    
    def __init__(
            self, 
            in_features,
            out_features, 
            use_bias = True    # 是否使用偏置
        ) -> None:
        np.random.seed(1)
        self.ltype = "Linear"
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        weight_scale = 1/in_features
        weight_shape = (self.out_features, self.in_features)
        self.weight = {
            "value": (np.random.randn(self.out_features, self.in_features) * np.sqrt(2 / self.in_features)).astype('float64'),
            # "value": (np.random.uniform(-weight_scale, weight_scale,weight_shape)).astype('float64'),
            "grad": np.zeros(weight_shape)
        }

        bias_shape = (self.out_features,)
        self.bias = {
            "value": np.zeros(bias_shape),
            # "value": (np.random.randn(self.out_features,)* np.sqrt(2 / self.in_features)).astype('float64'),
            # "value": (np.random.uniform(-weight_scale, weight_scale, bias_shape)).astype('float64'),
            "grad": np.zeros(bias_shape)
        }

        self.cache = None

    
    def forward(self, x):
        '''全连接层的前向计算.
        Args:
            x: 全连接层的输入, nd.array, shape = (N, IF)
        Returns:
            self.out: 全连接层的输出, nd.array, shape = (N, OF)
        '''
        self.cache = x

        self.out = np.dot(x, self.weight['value'].T)
        if self.use_bias:
            self.out = self.out+self.bias['value']
        # if self.use_bias:
        #     self.out = linear(x, self.weight['value'], self.bias['value'])
        # else:
        #     self.out = linear(x, self.weight['value'])

        return self.out


    def backward(self, dout):
        '''全连接层的反向传播
        Args:
            dout: Loss对输出out求导. np.array, shape = (N, OF)
        
        '''
        x = self.cache

        self.weight['grad'] = np.dot(dout.T, x)
        self.bias['grad'] = np.sum(dout, axis=0)
        # print("dout: {}, self.weight['value']: {}".format(dout.shape, self.weight['value'].shape))
        x_grad = np.dot(dout, self.weight['value'])
        
        return x_grad, self.weight['grad'], self.bias['grad']
    

def check_gradient(x, w, b, dx, dw, db, dout):
    import torch
    '''传入变量为nd.array
    '''
    x = torch.tensor(x, requires_grad=True)
    # w = torch.tensor(w)
    w = torch.tensor(w, requires_grad=True)
    dout = torch.tensor(dout, requires_grad=False)
    if b is None:
        y = torch.matmul(x, w.T)
    else:
        b = torch.tensor(b, requires_grad=True)
        y = torch.matmul(x, w.T)+b
    
    
    y.backward(dout)
    
    print("diff grad x: ", np.mean(np.abs(dx-x.grad.numpy())))
    print("diff grad w: ", np.mean(np.abs(dw-w.grad.numpy())))
    if b is not None:
        print("diff grad b: ", np.mean(np.abs(db-b.grad.numpy())))


if __name__ == "__main__":
    import torch.nn as nn
    import torch

    use_bias = False
    N = 10
    in_num = 20
    out_num = 30

    input = np.random.normal(0, 1, (N, in_num)).astype('float64')
    dout = np.random.rand(N, out_num)
    # print(input)

    ## numpy实现的卷积输出
    fc = Linear(in_features=in_num, out_features=out_num, use_bias=use_bias)
    y = fc.forward(input)
    x_grad, w_grad, b_grad = fc.backward(dout)

    w = fc.weight["value"]
    if fc.use_bias:
        b = fc.bias["value"]
    else:
        b = None
    check_gradient(input, w, b, x_grad, w_grad, b_grad, dout)
    # import torch.nn as nn
    # import torch

    # use_bias = True
    # N = 10
    # in_num = 20
    # out_num = 30

    # input = np.random.normal(0, 1, (N, in_num)).astype('float64')
    # dout = np.random.rand(N, out_num)
    # # print(input)

    # ## numpy实现的卷积输出
    # fc = Linear(in_features=in_num, out_features=out_num, use_bias=use_bias)
    # y = fc.forward(input)
    # _, w_grad, b_grad = fc.backward(dout)

    # pytorch卷积输出
    x_tensor = torch.Tensor(input).double()
    # w_tensor = torch.Tensor(conv.weight['value'])
    # b_tensor = torch.Tensor(conv.bias['value'])
    # # print("b_tensor: ", b_tensor)
    fc_pt = nn.Linear(in_num, out_num, bias=use_bias)
    fc_pt.weight = nn.Parameter(torch.DoubleTensor(fc.weight['value']))
    # fc_pt.bias = nn.Parameter(torch.DoubleTensor(fc.bias['value']))
    y_tensor = fc_pt(x_tensor)
    # y_tensor.backward(torch.Tensor(dout))
    
    # # print("y: {}\ny_tensor: {}".format(y, y_tensor))
    
    print("diff y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    # print("diff grad w: ", np.mean(np.abs(w_grad-fc_pt.weight.grad.numpy())))
    # print("diff grad b: ", np.mean(np.abs(b_grad-fc_pt.bias.grad.numpy())))