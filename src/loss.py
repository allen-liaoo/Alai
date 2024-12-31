import numpy as np
from . import utils as utils

# logits: (...1, Class, ...2)
# targets: (...1, C, ...2)
# loss: (...1, ...2)
# dl_dx: (...1, C, ...2)
def CrossEntropySoftmax(logits, targets, classAxis=0):
    # consulted https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    s = utils.softmax(logits, axis= classAxis) # sum across class dimension
    loss = -np.sum(targets * np.log(s), axis= classAxis)
    dl_dx = s - targets
    return loss, dl_dx