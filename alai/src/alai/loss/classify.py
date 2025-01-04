import numpy as np
from .loss import Loss
from .. import utils

class BinaryCrossEntropy(Loss):
    # ouputs and targets shape are the same
    # ouputs are tensors in {0, 1}
    @staticmethod
    def validateArgs(output, targets, **kwargs):
        if output.shape != targets.shape:
            raise ValueError(f"output {output.shape} and targets {targets.shape} should have same shape and targets")
        if ((output > 1) | (output < 0)).any() or ((targets > 1) | (targets < 0)).any():
            raise TypeError("output or target should be in the range [0, 1]")

    @utils.validate(validateArgs)
    @Loss.reducer(default= 'mean', axis= None)
    def forward(self, output, targets, **kwargs):
        # keep logits and (1-logits) in (0, 1]
        output = output.astype(float)
        offset = 1e-8
        ouput_ = np.clip(output, a_min= offset, a_max= 1) # in (0, 1]
        output__ = np.clip(output, a_min= 0, a_max= (1 - offset)) # in [0, 1) so that 1 - output__ is in (0, 1]

        loss = targets * np.log(ouput_) + (1 - targets) * np.log(1 - output__)
        loss = -1 * loss
        return loss

    @utils.validate(validateArgs)
    @Loss.reducer(default= 'none', axis= None)
    def backward(self, output, targets, **kwargs):
        # Make sure no division by 0
        offset = 1e-8
        ouput_ = np.clip(output, a_min= offset, a_max= 1-offset) # in (0, 1]
        return (ouput_ - targets) / (ouput_ *  (1 - ouput_))

class CrossEntropySoftmax(Loss):
    # logits: (...1, Class, ...2)
    # targets: (...1, C, ...2)
    @staticmethod
    def validateArgs(logits, targets, classAxis, **kwargs):
        if logits.shape != targets.shape:
            raise ValueError(f"output {logits.shape} and targets {targets.shape} should have same shape")
        if len(logits.shape) <= classAxis:
            raise ValueError(f"logits and targets should have at least classAxis+1 (={classAxis+1}) dimensions")

    # loss: (...1, ...2)
    @Loss.reducer(default= 'mean', axis= None)
    @utils.validate(validateArgs)
    def forward(self, logits, targets, classAxis= 0, **kwargs):
        s = utils.softmax(logits, axis= classAxis) # sum across class dimension
        return - np.sum(targets * np.log(s), axis= classAxis)

    # dl_dlogits: (...1, C, ...2)
    @Loss.reducer(default= 'none', axis= None)
    @utils.validate(validateArgs)
    def backward(self, logits, targets, classAxis= 0, **kwargs):
        # consulted https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        s = utils.softmax(logits, axis= classAxis) # sum across class dimension
        return s - targets