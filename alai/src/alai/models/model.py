from abc import ABC, abstractmethod
from typing import Tuple
from ..loss import Loss

class Model:
    def __init__(self, *, lossFn:Loss = None):
        self.lossFn = lossFn

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, input, targets):
        output = self.forward(input)
        loss, dl_doutput = self.compute_loss(output, targets)
        self.backward(dl_doutput)
        return loss

    # Forward pass through the network
    @abstractmethod
    def forward(self, input):
        pass

    # Backward pass through the network
    # lr: learning rate
    # usually void
    @abstractmethod
    def backward(self, dl_doutput, * lr):
        pass

    # Returns (loss, dl_doutput)
    def compute_loss(self, output, targets):
        return self.lossFn(output, targets)

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return self.__repr__()