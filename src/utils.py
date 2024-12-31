import numpy as np

# input (x) and returned tensor have the same dimensions
# axis: sum dimension
def softmax(x, *, axis=-1):
    e_x = np.exp(x - np.max(x))  # Subtract max to avoid overflow
    return e_x / np.sum(e_x, axis=axis, keepdims=True)