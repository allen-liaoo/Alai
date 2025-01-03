import numpy as np
from tqdm import tqdm 
from alai import layers

# Assumes one extra dimension B, batch size
def forward_iter(layer, x):
    y = np.zeros((x.shape[0], layer.out_dim))
    for b in range(x.shape[0]):
        for i in range(layer.out_dim):
            for j in range(layer.in_dim):
                y[b, i] += layer.w[i, j] * x[b, j]
            if layer.bias:
                y[b, i] += layer.b[i]
    return y

# Assumes one extra dimension B, batch size
def backward_iter(layer, dl_dy):
    dl_dx = np.zeros((dl_dy.shape[0], layer.in_dim)) # (B, I)
    for b in range(dl_dy.shape[0]):
        for j in range(layer.in_dim):
            for i in range(layer.out_dim):
                dl_dx[b, j] += layer.w[i, j] * dl_dy[b, i]

    dl_dw = np.zeros(layer.w.shape)
    for i in range(layer.out_dim):
        for j in range(layer.in_dim):
            for b in range(dl_dy.shape[0]):
                dl_dw[i, j] += dl_dy[b, i] * layer.cache.x[b, j]
            dl_dw[i,j] /= dl_dy.shape[0] # mean

    if layer.bias:
        dl_db = np.mean(dl_dy, axis= 0)[:, None] # (B, O) -> (O,) -> (O, 1)
        # dl_db = np.sum(dl_dy, axis= 0)[:, None]
    else:
        dl_db = None

    return dl_dx, dl_dw, dl_db

epoch = 30
for i in tqdm(range(epoch)):
    batch_size = 30
    linear = layers.Linear(50, 100)
    linear.forward(np.random.randn(batch_size, 50))
    dl_dy = np.random.randn(batch_size, 100)
    dl_dz, dl_dw, dl_db = backward_iter(linear, dl_dy)
    dl_dz2, dl_dw2, dl_db2 = linear.backward(dl_dy, update=False)

    assert np.allclose(dl_dz, dl_dz2)
    assert np.allclose(dl_dw, dl_dw2)
    assert np.allclose(dl_db, dl_db2)