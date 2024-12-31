import numpy as np
from typing import Literal
from . import utils as utils

processType = Literal['mean', 'sum', 'none']

# process loss and dl by takeing average or summing over a batch (or other extra dimensions)
# axis=None means averaging or summing over the whole tensor
def postProcess(loss, process: processType, axis: int|tuple[int, ...]|None):
    if process == 'mean':
        return np.mean(loss, axis= axis)
    elif process == 'sum':
        return np.sum(loss, axis= axis)
    else:
        return loss

# ouputs are tensors in {0, 1}
# ouputs and targets shape are the same
def BCE(logits, targets, *, process:processType = 'mean'):
    y = - targets * np.log(logits) + (1 - targets) * np.log(1 - logits)
    dl_dlogits = - targets / logits + (1 - targets) / (1 - logits)
    return postProcess(y, process, None), postProcess(dl_dlogits, process, None)

# logits: (...1, Class, ...2)
# targets: (...1, C, ...2)
# loss: (...1, ...2)
# dl_dx: (...1, C, ...2)
def CrossEntropySoftmax(logits, targets, classAxis= 0, *, process:processType= 'mean'):
    # consulted https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    s = utils.softmax(logits, axis= classAxis) # sum across class dimension
    loss = -np.sum(targets * np.log(s), axis= classAxis)
    dl_dlogits = s - targets
    return postProcess(loss, process, None), postProcess(dl_dlogits, process, None)