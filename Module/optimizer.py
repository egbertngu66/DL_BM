import numpy as np


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self):

        if self.v is None:
            self.v = dict()
            for layer, val in self.params.items():
                self.v[layer] = dict()
                for name, w in val.items():
                    self.v[layer][name] = np.zeros_like(w['value'])
        
        for layer, value in self.params.items():
            for key in value.keys():
                self.v[layer][key] = self.momentum * self.v[layer][key] + (1 - self.momentum) * self.params[layer][key]['grad']
                self.params[layer][key]['value'] -= self.lr * self.v[layer][key]


    def state_dict(self):
        state_dict = dict()
        state_dict = {
            'lr': self.lr,
            'momentum': self.momentum,
            'v': self.v
        }

        return state_dict
    

    def load_state_dict(self, state_dict):
        
        self.lr = state_dict['lr']
        self.momentum = state_dict['momentum']
        self.v = state_dict['v']
