from typing import Literal
import numpy as np
from abc import ABC, abstractmethod

## REDUCTION ##
# In many cases (loss functions and backpropagation), we need to reduce the output of the function to a scalar or lower dimensional tensor
# often because the input has a batch dimension (or other reasons).
# This module provides a decorator to reduce the output of a function.
reductionType = Literal['mean', 'sum', 'none']

# Reduce tensor by taking average or summing over specified axis (None if over the whole tensor)
def reduce(tensor, reduction: reductionType, axis: int|tuple[int, ...]|None):
    if reduction == 'mean':
        return np.mean(tensor, axis= axis)
    elif reduction == 'sum':
        return np.sum(tensor, axis= axis)
    else:
        return tensor

# How to use:
# lossFunc = Loss(reduction=...)
# loss, dl_doutput = lossFunc(output, targets)
# How to extend: implement forward() and backward(), and optionally, validateArgs()
class Loss(ABC):
    def __init__(self):
        pass

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

    # decorator to reduce certain output of the function via reduce() method
    # axis: the axis(es) to reduce
    # target: the location of the tensor to reduce. for tuples of outputs, this is the index. Default is None, which is the ouput itself (as a tensor)
    @staticmethod
    def reducer(func=None, *, default:reductionType= 'none', axis: int|tuple[int, ...]|None= None, target: int|None= None):
        def decorator(func):
            def newFunc(*args, **kwargs):
                ret = func(*args, **kwargs)

                reduction = default
                if 'reduction' in kwargs:
                    reduction = kwargs['reduction']

                if target is not None: # ret is a tuple, and target is location of outout to reduce
                    ret = list(ret)
                    ret[target] = reduce(ret, reduction, axis)
                    ret = tuple(ret)
                else: # reduce the whole output
                    ret = reduce(ret, reduction, axis)
                return ret
            return newFunc

        # allows @decorator or @decorator(...)
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