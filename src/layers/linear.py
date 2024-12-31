from .layer import Layer
from typing import Literal
import numpy as np

class Linear(Layer):
    # w: (Out, In)
    # b: (O, 1)
    def __init__(self, in_dim, out_dim, bias=True):
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

    # dl_dy: (..., O)
    # dl_dx: (..., I)
    # dl_dw: (..., O, I)
    # dl_db: (..., O)
    # "..." denotes the same shape
    def backward(self, dl_dy):
        # for each number x_i of the last dimension of x,
        # the number was multiplied with the ith column of w
        # and by chain rule, each jth row of w must multiply (dl_dy)_j as a scalar (by elementwsise multiply then sum across rows / O dim)
        dl_dx = self.w * dl_dy[...,None] # extend dl_dy by a new dimension for broadcasting last dim: (O, I) and (..., O, 1)
        dl_dx = np.sum(dl_dx, axis=-2) # sum across O dim

        # outer product, ∂w_ij = ∂y_i * x_j
        dl_dw = (dl_dy[..., None]   # change dl_dy to (..., O, 1)
            @ self.x[..., None, :]) # change x to (..., 1, I) for broadcasting with dl_dy

        if self.bias:
            dl_db = dl_dy
        else:
            dl_db = None
        return dl_dx, dl_dw, dl_db

    # Treat all extra dimensions as batches
    # So if dl_dw is (..., O, I), and ... is (x, y, z), then there are x*y*z batches
    # sum method simply sums the gradient across batches, whereas mean methid takes the average
    def update_weights(self, learning_rate, dl_dw, dl_db=None, *, method:Literal['sum','mean']='mean'):
        batch_size = np.prod(dl_dw.shape[:-2]) if method == 'mean' else 1
        axes_to_sum = tuple(x for x in range(dl_dw.ndim-2))

        dl_dw = np.sum(dl_dw, axis= axes_to_sum) # sum across batch
        self.w -= learning_rate * dl_dw / batch_size

        if self.bias:
            dl_db = np.sum(dl_db, axis= axes_to_sum)
            self.b -= learning_rate * dl_db / batch_size