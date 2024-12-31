import numpy as np
from typing import Literal
from .layer import Layer

class Linear(Layer):
    # w: (Out, In)
    # b: (O, 1)
    def __init__(self, in_dim, out_dim, *, bias=True):
        self.w = np.random.randn(out_dim, in_dim)
        self.b = np.zeros((out_dim, 1)) if bias else None
        self.bias = bias
        self.x = None

    # x: (..., I)
    # returned: (..., O)
    # "..." denotes the same shape
    # for multi-dimensional x, only last dimension are multiplied with w. other dimensions are broadcasted
    def forward(self, x):
        self.x = x

        # change x to (..., I, 1) to handle broadcasting (O, I) @ (..., I, 1), 
        # so that w is multiplied by each (I, 1) tensor in batch, producing (..., O, 1)
        x = x[..., None]
        y = self.w @ x
        if self.bias:
            y = y + self.b      # uses broadcasting of (O, 1) to (B, O, 1) across batch
        return y[..., 0] # get rid of last dimension

    # TODO: This function is generalized to recieve any n-dim input, but it is unnecessary as
    # loss functions are now designed to eliminate extra dimensions
    # dl_dy: (..., O)
    # dl_dx: (..., I)
    # dl_dw: (O, I)
    # dl_db: (O, 1)
    # "..." denotes the same shape
    def backward(self, dl_dy, *, lr=1e-3, update=True):
        # for each number x_i of the last dimension of x,
        # the number was multiplied with the ith column of w
        # and by chain rule, each jth row of w must multiply (dl_dy)_j as a scalar (by elementwsise multiply then sum across rows / O dim)
        dl_dx = self.w * dl_dy[...,None] # extend dl_dy by a new dimension for broadcasting last dim: (O, I) and (..., O, 1)
        dl_dx = np.sum(dl_dx, axis=-2) # sum across O dim

        # outer product, ∂w_ij = ∂y_i * x_j
        dl_dw = (dl_dy[..., None]   # change dl_dy to (..., O, 1)
            @ self.x[..., None, :]) # change x to (..., 1, I) for broadcasting with dl_dy
        dl_dw = np.mean(dl_dw, axis= tuple(range(dl_dw.ndim - 2))) # (..., O, I) -> (O, I), if ... is empty, dl_dw unchanged

        if self.bias:
            dl_db = np.mean(dl_dy, axis= tuple(range(dl_dy.ndim - 1)))[:, None] # (..., O) -> (O,) -> (O, 1)
        else:
            dl_db = None

        if update:
            self.update_weights(dl_dw, dl_db, lr=lr)
        return dl_dx, dl_dw, dl_db

    def update_weights(self, dl_dw, dl_db, *, lr):
        self.w -= lr * dl_dw

        if self.bias and dl_db is not None:
            self.b -= lr * dl_db