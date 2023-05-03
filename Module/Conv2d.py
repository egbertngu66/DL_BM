import numpy as np
# @TODO stride不为1的时候存在bug


class Conv2d(object):
    def __init__(
            self, 
            shape,          # shape = (N, C, H, W)
            output_channels,
            ksize=3, 
            stride=1, 
            padding=0, 
            use_bias = False    # 是否使用偏置
        ) -> None:
        self.ltype = "Conv"
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[1]
        self.batch_size = shape[0]
        self.stride = stride
        if type(ksize) == int:
            ksize = (ksize, ksize)
        self.ksize = ksize
        self.padding = padding
        self.use_bias = use_bias

        weight_scale = np.sqrt(ksize[0]*ksize[1]*self.input_channels/2)      # @TODO MSRA方法, 加速收敛
        # weight shape = (OC, IC, kH, kW); bias shape = (OC, )
        weight_shape = (self.output_channels, self.input_channels, ksize[0], ksize[1])
        self.weight = {
            "value": (np.random.standard_normal(weight_shape) / weight_scale).astype('float64'),
            "grad": np.zeros(weight_shape)
        }

        bias_shape = (self.output_channels,)
        self.bias = {
            "value": (np.random.standard_normal(bias_shape) / weight_scale).astype('float64'),
            "grad": np.zeros(bias_shape)
        }

        self.output_shape = (
            self.batch_size,
            self.output_channels,
            int((shape[2]+2*self.padding - ksize[0]) / self.stride + 1),
            int((shape[3]+2*self.padding - ksize[1]) / self.stride + 1)     
        )
        
        self.cache = None
        # # eta 存放 backward时Loss对Layer out的求导
        # self.eta = np.zeros(self.output_shape)

        # self.w_gradient = np.zeros(self.weight.shape)
        # self.b_gradient = np.zeros(self.bias.shape)

    
    def forward(self, x):
        '''卷积层的前向计算.
        Args:
            x: 卷积的输入, nd.array, shape = (N, IC, IH, IW)
        Returns:
            self.out: 卷积结果, nd.array, shape = (N, OC, OH, OW)
        '''
        # self.batch_size = x.shape[0]
        N, OC, OH, OW = self.output_shape
        self.out = np.zeros(self.output_shape)
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        
        for n in range(self.batch_size):
            for c in range(OC):
                for i in range(OH):
                    h_start = i*self.stride
                    h_end = h_start+self.ksize[0]
                    for j in range(OW):
                        w_start = j*self.stride
                        w_end = w_start+self.ksize[1]
                        window = x[n, :, h_start:h_end, w_start:w_end]
                        # print("n = {}; c = {}; i = {}; j = {}".format(n,c,i,j))
                        # print(np.sum(window*self.weight['value'][c]))
                        self.out[n, c, i, j] = np.sum(window*self.weight['value'][c])

        
        # weight_cols = self.weight.reshape((self.output_channels, -1))  # shape = (OC, IC*KW*KH)
        # x_cols = im2col(x, self.ksize, self.stride, self.pad)
        # out_cols = np.dot(x_cols,weight_cols.T)  # shape = (N*OH*OW, OC)
        # # print("out_cols: {}\nself.bias :{}".format(out_cols, self.bias))
        if self.use_bias == True:
            self.out = self.out + self.bias['value'].reshape(1, self.output_channels, 1, 1)
        
        self.cache = x
        return self.out


    def backward(self, dout):
        '''卷积层的反向传播
        Args:
            dout: Loss对输出out求导. np.array, shape = (N, OC, OH, OW)
        
        '''
        x = self.cache
        # x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        N, IC, IH, IW = x.shape
        N, OC, OH, OW = dout.shape
        x_grad = np.zeros(x.shape)

        # weight求导
        for c1 in range(OC):
            for c2 in range (IC):
                for i in range(self.ksize[0]):
                    h_start = i*self.stride
                    h_end = h_start+OH
                    for j in range(self.ksize[1]):
                        w_start = j*self.stride
                        w_end = w_start+OW
                        window = x[:, c2, h_start:h_end, w_start:w_end]
                        # print(dout[:, c1].shape, window.shape)
                        # print("dout[n,c, i, j]:\n{}\nx[n, :, h_start:h_end, w_start:w_end]:\n{}".format(dout[n, c, i, j], x[n, :, h_start:h_end, w_start:w_end]))
                        self.weight['grad'][c1, c2, i, j] = np.sum(dout[:, c1] * window)
        # bias 求导
        for c in range(OC):
            self.bias['grad'][c] = np.sum(dout[:, c])

        # x求导
        padding_size = (
            int(((IH-1)*self.stride+KH-OH)/2),
            int(((IW-1)*self.stride+KW-OW)/2)
        )
        # print(padding_size)
        dout_pad = np.pad(dout, ((0, 0), (0, 0), padding_size, padding_size), 'constant', constant_values=0)
        filter_flip = self.weight['value'][:, :, ::-1, ::-1]    # shape = (OC, IC, KH, KW)
        # # print("filter:\n:{}\nfilter_flipped:\n:{}".format(self.weight['value'], filter_flip))
        for n in range(N):
            for c in range(IC):
                for i in range(IH):
                    h_start = i*self.stride
                    h_end = h_start+KH
                    for j in range(IW):
                        w_start = j*self.stride
                        w_end = w_start+KW
                        window = dout_pad[n, :, h_start:h_end, w_start:w_end]
                        # print(i, j)
                        # print("window: {}".format(window))
                        x_grad[n, c, i ,j] = np.sum(filter_flip[:, c, ...]*window)
        
        # x_grad_t = np.zeros(x.shape)
        # for n in range(N):
        #     for c in range (OC):
        #         for i in range(OH):
        #             h_start = i*self.stride
        #             h_end = h_start+self.ksize[0]
        #             for j in range(OW):
        #                 w_start = j*self.stride
        #                 w_end = w_start+self.ksize[1]
        #                 # print("dout[n,c, i, j]:\n{}\nx[n, :, h_start:h_end, w_start:w_end]:\n{}".format(dout[n, c, i, j], x[n, :, h_start:h_end, w_start:w_end]))
        #                 # self.weight['grad'][c,...] += dout[n,c, i, j] * x[n, :, h_start:h_end, w_start:w_end]
        #                 x_grad_t[n, :, h_start:h_end, w_start:w_end] += dout[n, c, i, j] * self.weight['value'][c, ...]
        
        # print(x_grad, x_grad_t)
        # print("diff grad x: ", np.sum(x_grad-x_grad_t))
        return x_grad, self.weight['grad'], self.bias['grad']


def check_gradient(x, w, b, dx, dw, db, dout, param):
    import torch
    '''传入变量为nd.array
    '''
    # w = torch.tensor(w)
    w = torch.tensor(w, requires_grad=True)
    dout = torch.tensor(dout, requires_grad=False)

    N, OC, OH, OW = dout.size()
    y = torch.zeros(dout.size(), requires_grad=True)
    x = np.pad(x, ((0, 0), (0, 0), (param['pad'], param['pad']), (param['pad'], param['pad'])), 'constant', constant_values=0)
    x = torch.tensor(x, requires_grad=True)

    for n in range(N):
        for c in range(OC):
            for i in range(OH):
                h_start = i*param["stride"]
                h_end = h_start+param["ksize"][0]
                for j in range(OW):
                    w_start = j*param["stride"]
                    w_end = w_start+param["ksize"][1]
                    window = x[n, :, h_start:h_end, w_start:w_end]
                    # print("n = {}; c = {}; i = {}; j = {}".format(n,c,i,j))
                    # print(np.sum(window*self.weight['value'][c]))
                    y_clone = y.clone()
                    y_clone[n, c, i, j] = torch.sum(window*w[c])
                    y = y_clone


    if b is not None:
        b = torch.tensor(b, requires_grad=True)
        y = y + b  
    
    y.backward(dout)
    
    print("diff grad x: ", np.mean(np.abs(dx-x.grad.numpy())))
    print("diff grad w: ", np.mean(np.abs(dw-w.grad.numpy())))
    if b is not None:
        print("diff grad b: ", np.mean(np.abs(db-b.grad.numpy())))


if __name__ == "__main__":
    import torch.nn as nn
    import torch

    use_bias = True
    N, IC, IH, IW = 1, 1, 4, 4
    KH, KW = 3, 3
    stride = 1
    padding = 1
    OC = 1
    OH = int((IH+2*padding -KH) / stride + 1)
    OW = int((IW+2*padding - KH) / stride + 1)

    image = np.random.normal(0, 1, (N, IC, IH, IW)).astype('float64')
    dout = np.random.rand(N, OC, OH, OW)
    # print(image)

    ## numpy实现的卷积输出
    conv = Conv2d(image.shape, OC, (KH, KW), stride, padding=padding,use_bias=use_bias)
    y = conv.forward(image)
    x_grad, w_grad, b_grad = conv.backward(dout)
    w = conv.weight["value"]
    if conv.use_bias:
        b = conv.bias["value"]
    else:
        b = None
    
    param = {
        "pad": padding,
        "stride": stride,
        "ksize": [KH, KW]
    }
    check_gradient(image, w, b, x_grad, w_grad, b_grad, dout, param)

    # # pytorch卷积输出
    # x_tensor = torch.Tensor(image).double()
    # # w_tensor = torch.Tensor(conv.weight['value'])
    # # b_tensor = torch.Tensor(conv.bias['value'])
    # # print("b_tensor: ", b_tensor)
    # conv_pt = nn.Conv2d(IC, OC, (KH, KW), stride=stride, padding=padding, bias=use_bias)
    # conv_pt.weight = nn.Parameter(torch.DoubleTensor(conv.weight['value']))
    # conv_pt.bias = nn.Parameter(torch.DoubleTensor(conv.bias['value']))
    # y_tensor = conv_pt(x_tensor)
    # y_tensor.backward(torch.Tensor(dout))
    
    # # print("y: {}\ny_tensor: {}".format(y, y_tensor))
    
    # print("diff y: ", np.mean(np.abs(y-y_tensor.detach().numpy())))
    # print("diff grad w: ", np.mean(np.abs(w_grad-conv_pt.weight.grad.numpy())))
    # print("diff grad b: ", np.mean(np.abs(b_grad-conv_pt.bias.grad.numpy())))