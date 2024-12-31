import numpy as np
from .layer import Layer

class Sigmoid(Layer):
    def __init__(self):
        self.y = None

    # element-wise sigmoid
    # x: (...)
    # y: (...)
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    # dl_dy same dimension as dl_dx and x
    # dl_dy: (...)
    # dl_dx: (...)
    def backward(self, dl_dy):
        dl_dx = dl_dy * self.y * (1 - self.y)
        return dl_dx

    def update_weights(self):
        pass