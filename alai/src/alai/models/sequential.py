from typing import List
from .model import Model
from ..layers import Layer

class Sequential(Model):
    def __init__(self, layers:List[Layer]= [], **kwargs):
        super().__init__(**kwargs)
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

    def backward(self, dl_doutput, *, lr= 1e-3):
        for layer in reversed(self.layers):
            # Check if layer.backward is marked as is_updatable by @updatable
            if getattr(layer.backward, 'is_updatable', False):
                dl_doutput = layer.backward(dl_doutput, update= True, lr= lr)
            else:
                dl_doutput = layer.backward(dl_doutput)

            if isinstance(dl_doutput, tuple):
                dl_doutput = dl_doutput[0] # the rest are dl_dparams

    def __repr__(self):
        return f"{super().__repr__()} ({self.layers})"

    def __str__(self):
        return self.__repr__()