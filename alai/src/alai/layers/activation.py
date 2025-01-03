import numpy as np
from .layer import Layer

class Activation(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_weights(self):
        raise NotImplementedError("update_weights not implemented for activation layers")

class Relu(Activation):
    def __init__(self):
        super().__init__()

    # element-wise relu
    # x: (...)
    # y: (...)
    def forward(self, x):
        y = np.where(x < 0, 0, x)
        self.cache.x = x
        return y

    # dl_dy: (...)
    # dl_dx: (...)
    def backward(self, dl_dy):
        dl_dx = np.where(self.cache.x < 0, 0, dl_dy)
        return dl_dx

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    # element-wise sigmoid
    # x: (...)
    # y: (...)
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.cache.y = y
        return y

    # dl_dy: (...)
    # dl_dx: (...)
    def backward(self, dl_dy):
        dl_dx = dl_dy * self.cache.y * (1 - self.cache.y)
        return dl_dx