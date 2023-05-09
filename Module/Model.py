import numpy as np


class Model:

    def __init__(self) -> None:
        
        # self.isTrain = True
        # self.lr = lr
        self.layers_dict = None


    # def forward(self, x):

    #     for layer in self.layers:
    #         print("x shape: {}".format(x.shape))
    #         out = layer.forward(x)
    #         x = out

    #     return out
    
    def backward(self, dout):
        '''
        Args:
            out: 网络的最终输出
            dout: loss对out的偏导数
        '''
        for layer in list(self.layers_dict.values())[::-1]:
            # print("layer.ltype: {}, dout shape: {}".format(layer.ltype, dout.shape))
            dout = dout.reshape(layer.out.shape)
            # print(dout.shape)
            if layer.ltype == "Activation":
                dout = layer.backward(dout)
            elif layer.ltype == "Linear" or layer.ltype == "Conv2d":
                dout, dw, db = layer.backward(dout)
                # print("dw: {}\ndb: {}".format(db, dw))
                layer.weight["grad"] = dw
                layer.bias["grad"] = db
                # layer.weight["value"] -= dw * self.lr
                # layer.bias["value"] -= db * self.lr
            elif layer.ltype == "BatchNorm":
                dout, dgamma, dbeta = layer.backward(dout)
                layer.gamma['grad'] = dgamma
                layer.beta['grad'] = dbeta
                # layer.gamma['value'] -= dgamma * self.lr
                # layer.beta['value'] -= dbeta * self.lr
    

    def load_state_dict(self, state_dict):
        '''
        state_dict: dict, key为layer名称, value为dict
        '''
        for key, value in state_dict.items():
            if key in self.layers_dict.keys():
                for w_key, w_value in value.items():
                    # print(key, w_key, w_value)
                    layer_weight = getattr(self.layers_dict[key], w_key)
                    if w_key == 'running_mean' or w_key == 'running_var':
                        layer_weight = w_value
                    else:
                        layer_weight['value'] = w_value
                    # self.layers_dict[key][w_key] = w_value
                    setattr(self.layers_dict[key], w_key, layer_weight)

    def train(self):
        for layer in self.layers_dict.values():
            if layer.ltype == "BatchNorm":
                layer.isTrain = True
    

    def eval(self):
        for layer in self.layers_dict.values():
            if layer.ltype == "BatchNorm":
                layer.isTrain = False


    def parameters(self):
        params = dict()
        for name, layer in self.layers_dict.items():
            
            if layer.ltype == "Linear" or layer.ltype == "Conv2d":
                params[name] = dict()
                params[name]["weight"] = layer.weight
                params[name]["bias"] = layer.bias
            elif layer.ltype == "BatchNorm":
                params[name] = dict()
                params[name]["gamma"] = layer.gamma
                params[name]["beta"] = layer.beta
        # for layer in self.layers_dict.values():
        #     layer.weight["value"] -= dw * self.lr
        #     layer.bias["value"] -= db * self.lr

        return params
    

    def state_dict(self):
        state_dict = dict()
        for name, layer in self.layers_dict.items():
            
            if layer.ltype == "Linear" or layer.ltype == "Conv2d":
                state_dict[name] = dict()
                state_dict[name]["weight"] = layer.weight['value']
                state_dict[name]["bias"] = layer.bias['value']
            elif layer.ltype == "BatchNorm":
                state_dict[name] = dict()
                state_dict[name]["gamma"] = layer.gamma['value']
                state_dict[name]["beta"] = layer.beta['value']
                state_dict[name]["running_mean"] = layer.running_mean
                state_dict[name]["running_var"] = layer.running_var

        return state_dict