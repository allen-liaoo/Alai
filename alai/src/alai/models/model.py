from abc import ABC, abstractmethod

class Model:
    def __init__(self):
        pass

    # Forward pass through the network
    # Computes and return loss and derivative of loss function with respect to the output
    # usually returns (Output, Loss, dl_dOutput)
    @abstractmethod
    def forward(self):
        pass

    # Backward pass through the network
    # lr: learning rate
    # update: if True, updates each layer's weights
    # usually returns None
    @abstractmethod
    def backward(self, *, lr, update):
        pass