import numpy as np
from .linear import Linear

class Embedding(Linear):
    # w: (Embed, Vocab) - each column is the embedding of a token
    def __init__(self, vocab_size, embed_dim):
        super().__init__(in_dim= vocab_size, out_dim= embed_dim, bias= False)
        self.vocab_size = vocab_size

    # idxs: (B, N) - each element is an index of a token
    # y: (B, N, E)
    def forward(self, idxs):
        # transform (B, N) of row-indexes to (B, N, E) of one-hot encodings
        B, N = idxs.shape
        one_hots = np.zeros((B, N, self.vocab_size))
        b_indices, n_indices = np.meshgrid(np.arange(B), np.arange(N), indexing= 'ij')
        one_hots[b_indices, n_indices, idxs] = 1
        return super().forward(one_hots)

    # dl_dy: (B, N, E)
    # dl_dw: (B, V, E) - each batch is a matrix W with a scalar from dl_dy at every index used in the batch, and 0 elsewjere
    def backward(self, dl_dy, *, lr= 1e-3, update= True):
        dl_dx, dl_dw, _ = super().backward(dl_dy, lr= lr, update= update)
        return dl_dx, dl_dw

# Embed tokens (integers) into real-valued tensors
# Basically a linear layer that uses indexing instead of matrix multiplication
# class Embedding(Layer):
#     # w: (Vocab, Embed) - each row is the embedding of a token
#     def __init__(self, vocab_size, embed_dim):
#         self.w = np.random.randn(vocab_size, embed_dim)
#         self.batch_ids = None

#     # idxs: (Batch, N) - indexes of tokens, N per batch
#     # y: (B, N, E)
#     def forward(self, idxs):
#         self.batch_ids = idxs
#         B = idxs.shape[0]
#         E = self.w.shape[1]

#         idxs = np.reshape(idxs, (-1,)) # flatten idxs
#         y = self.w[idxs, :] # use idxs as row indexes, produces (B*N, E)
#         y = np.reshape(y, shape= (B, -1, E))
#         return y

#     # dl_dy: (B, N, E)
#     # dl_dw: (B, V, E) - each batch is a matrix W with a scalar from dl_dy at every index used in the batch, and 0 elsewjere
#     def backward(self, dl_dy):
#         B = dl_dy.shape[0]
#         V, E = self.w.shape

#         dl_dw = np.zeros((B, V, E))
#         for b in range(B):
#             dl_dw[b, self.batch_ids[b], :] = dl_dy[b, :, :]
#         return dl_dw

#     def update_weights(self, learning_rate, dl_dw):
#         batch_size = dl_dw.shape[0]
#         dl_dw = np.sum(dl_dw, axis= 0) # sum across batch
#         self.w -= learning_rate * dl_dw / batch_size