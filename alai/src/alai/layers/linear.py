import numpy as np
from .layer import Layer

class Linear(Layer):
    # w: (Out, In)
    # b: (O, 1)
    def __init__(self, in_dim, out_dim, *, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.register_params(
            w= np.random.random((out_dim, in_dim)),
            b= np.zeros((out_dim, 1), dtype= float) if bias else None
        )

    # x: (..., I)
    # y: (..., O)
    # "..." denotes the same shape
    # for multi-dimensional x, only last dimension are multiplied with w. other dimensions are broadcasted
    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Linear.forward: last dimension of x must be {self.in_dim}")

        self.cache.x = x

        # change x to (..., I, 1) to handle broadcasting (O, I) @ (..., I, 1), 
        # so that w is multiplied by each (I, 1) tensor, producing (..., O, 1)
        x = x[..., None]
        y = self.w @ x
        if self.bias:
            y = y + self.b      # uses broadcasting of (O, 1) to (B, O, 1) to add bias across other dimensions
        y = y[..., 0] # get rid of last dimension

        assert y.shape == self.cache.x.shape[:-1]+(self.out_dim,), f"Linear.forward: Expected y to have shape {self.x.shape[:-1]+(self.out_dim)} but got {y.shape}"
        return y

    # dl_dy: (..., O)
    # dl_dx: (..., I)
    # dl_dw: (O, I)
    # dl_db: (O, 1)
    # "..." denotes the same shape
    @Layer.auto_update
    def backward(self, dl_dy, **kwargs):
        if self.cache.x is None:
            raise ValueError("Linear.backward: forward must be called before calling backward")
        expected_shape = self.cache.x.shape[:-1] + (self.out_dim,)
        if dl_dy.shape != expected_shape:
            raise ValueError(f"Linear.backward: Expected dl_dy to have shape {expected_shape} but got {dl_dy.shape}")

        # for each number x_i of the last dimension of x,
        # the number was multiplied with the ith column of w
        # and by chain rule, each jth row of w must multiply (dl_dy)_j as a scalar (by elementwsise multiply then sum across rows / O dim)
        dl_dx = self.w * dl_dy[...,None] # extend dl_dy by a new dimension for broadcasting last dim: (O, I) and (..., O, 1)
        dl_dx = np.sum(dl_dx, axis=-2) # sum across O dim

        # outer product, ∂w_ij = ∂y_i * x_j
        dl_dw = (dl_dy[..., None]   # change dl_dy to (..., O, 1)
            @ self.cache.x[..., None, :]) # change x to (..., 1, I) for broadcasting with dl_dy
        dl_dw = np.mean(dl_dw, axis= tuple(range(dl_dw.ndim - 2))) # (..., O, I) -> (O, I), if ... is empty, dl_dw unchanged
        # dl_dw = np.sum(dl_dw, axis= tuple(range(dl_dw.ndim - 2)))

        if self.bias:
            dl_db = np.mean(dl_dy, axis= tuple(range(dl_dy.ndim - 1)))[:, None] # (..., O) -> (O,) -> (O, 1)
            # dl_db = np.sum(dl_dy, axis= tuple(range(dl_dy.ndim - 1)))[:, None]
        else:
            dl_db = None

        return dl_dx, dl_dw, dl_db

    def __repr__(self):
        return f"Linear (in={self.in_dim}, out={self.out_dim}, bias={self.bias})"
