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
    # Updates each layer's weights if update_weights is true
    # usually returns None
    @abstractmethod
    def backward(self, *, learning_rate, update_weights):
        pass