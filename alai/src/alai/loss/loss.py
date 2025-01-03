import numpy as np
from abc import ABC, abstractmethod
from alai import reduction

# How to use:
# lossFunc = Loss(reduction=...)
# loss, dl_doutput = lossFunc(output, targets)
# How to extend: implement forward() and backward(), and optionally, validateArgs()
class Loss(ABC):
    def __init__(self, reduction:reduction.reductionType= 'mean', reduceLoss:bool= True, reduceDl:bool= False):
        self.reduceLoss = reduceLoss
        self.reduceDl = reduceDl
        self.reduction = reduction

    def __call__(self, *args, calc_grad= True, **kwargs):
        loss = self.forward(*args, **kwargs)
        dl_doutput = self.backward(*args, **kwargs) if calc_grad else None
        return loss, dl_doutput

    @abstractmethod
    def forward(self, output, targets):
        pass

    @abstractmethod
    def backward(self, output, targets):
        pass

    # decorator for forward() that reduces the loss returned by forward(), using specified reduction type
    @staticmethod
    def fwd_reduce(func=None, *, axes: int|tuple[int, ...]|None= None):
        def decorator(func):
            def newFunc(self, *args, **kwargs):
                loss = func(self, *args, **kwargs)
                if self.reduceLoss:
                    loss = reduction.reduce(loss, self.reduction, axes)
                return loss
            return newFunc

        if func is None:
            return decorator
        else:
            return decorator(func)

    # decorator for backward() that reduces the dl returned by forward(), using specified reduction type
    @staticmethod
    def bwd_reduce(func=None, *, axes: int|tuple[int, ...]|None= None):
        def decorator(func):
            def newFunc(self, *args, **kwargs):
                dl = func(self, *args, **kwargs)
                if self.reduceDl:
                    dl = reduction.reduce(dl, self.reduction, axes)
                return dl
            return newFunc

        if func is None:
            return decorator
        else:
            return decorator(func)

class MeanSquaredError(Loss):
    @Loss.fwd_reduce
    def forward(self, output, targets):
        return np.mean((output - targets) ** 2, axis= None)

    @Loss.bwd_reduce
    def backward(self, output, targets):
        return 2 * (output - targets)