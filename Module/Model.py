import numpy as np


class Model:

    def __init__(self, lr) -> None:
        
        self.Training = True
        self.lr = lr
        self.layers = None


    def forward(self, x):

        for layer in self.layers:
            print("x shape: {}".format(x.shape))
            out = layer.forward(x)
            x = out

        return out
    
    def backward(self, dout):
        '''
        Args:
            out: 网络的最终输出
            dout: loss对out的偏导数
        '''
        for layer in self.layers[::-1]:
            # print("layer.ltype: {}, dout shape: {}".format(layer.ltype, dout.shape))
            dout = dout.reshape(layer.out.shape)
            if layer.ltype == "Activation":
                dout = layer.backward(dout)
            elif layer.ltype == "Linear" or layer.ltype == "Conv2d":
                dout, dw, db = layer.backward(dout)
                layer.weight["value"] -= dw * self.lr
                layer.bias["value"] -= db * self.lr
            elif layer.ltype == "BatchNorm":
                dout, dgamma, dbeta = layer.backward(dout)
                layer.gamma['value'] -= dgamma * self.lr
                layer.beta['grad'] -= dbeta * self.lr
                
