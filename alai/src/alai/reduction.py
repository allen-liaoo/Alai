# Status: Not used

# In many cases (loss functions and backpropagation), we need to reduce the output of the function to a scalar or lower dimensional tensor
# often because the input has a batch dimension (or other reasons).
# This module provides a decorator to reduce the output of a function.
from typing import Literal
import numpy as np

reductionType = Literal['mean', 'sum', 'none']

# Reduce tensor by taking average or summing over specified axis (None if over the whole tensor)
def reduce(tensor, reduction: reductionType, axis: int|tuple[int, ...]|None):
    if reduction == 'mean':
        return np.mean(tensor, axis= axis)
    elif reduction == 'sum':
        return np.sum(tensor, axis= axis)
    else:
        return tensor

# decorator to reduce certain output of the function via reduce() method
# requires that the function has a keyword parameter named "reduction" of type reductionTyoe
# axis: the axis(es) to reduce
# target: the location of the tensor to reduce. for tuples of outputs, this is the index. Default is None, which is the ouput itself (as a tensor)
def reducer(func=None, *, axis: int|tuple[int, ...]|None= None, target: int|None= None):
    def decorator(func):
        def newFunc(*args, **kwargs):
            ret = func(*args, **kwargs)
            reduction = kwargs['reduction']
            if target is not None: # ret is a tuple, and target is location of outout to reduce
                ret = list(ret)
                ret[target] = reduce(ret, reduction, axis)
                ret = tuple(ret)
            else:
                ret = reduce(ret, reduction, axis)
            return ret
        return newFunc

    # allows @decorator or @decorator(...)
    if func is None:
        return decorator
    else:
        return decorator(func)