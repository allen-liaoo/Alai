# TODO: Conform to breaking changes in alai API. also, nonfunctional prior to changes
import numpy as np
from ..layers.embedding import Embedding
from ..loss import loss as Loss
from .. import utils as utils

class BigramLanguageModel:
    def __init__(self, vocab_size):
        self.embed = Embedding(vocab_size= vocab_size, embed_dim= vocab_size)

    # idxs: (B, N) - batches of N tokens
    # targets: (B, N) – batches of N target tokens
    # logits: (B, N, E)
    # loss: (B, 1) – loss per batch
    # dl_dlogits: (B, N, E)
    def forward(self, idxs, targets= None):
        logits = self.embed.forward(idxs) # (B, N, E)

        if targets is None:
            return logits, None, None

        logits_t = np.transpose(logits, axes= (0, 2, 1)) # (B, E, N) for cross entropy
        targets = targets[:, None, :] # make targets (B, 1, N), broadcastable to logits

        # cross entropy expects classes to be second dimension
        loss, dl_dlogits = Loss.CrossEntropySoftmax(logits_t, targets, classAxis= 1) # (B, N), (B, E, N)
        loss = np.sum(loss, axis= 1)
        dl_dlogits = np.transpose(dl_dlogits, axes= (0, 2, 1))

        return logits, loss, dl_dlogits

    def backward(self, dl_dlogits, *, lr= 1e-3, update= True):
        self.embed.backward(dl_dlogits, lr=lr, update=update)

    # idxs: (N,) - sequence of N tokens
    # max_new_tokens: scalar
    def generate(self, idxs, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(idxs[None, :]) # idxs: (B=1, N), logits: (B, N, E)
            # take only the last token
            logits = logits[:, -1, :] # (B, E)

            # get probabilities by applying softmax
            # note: this part is very weird. Since E = V, we have V probabilities, 
            # which is what we want but doesn't seem to make sense if E != V
            probs = utils.softmax(logits) # probs: (B, E) = (B, V)

            # randomly get a next token from the list of probabilities
            id_next = np.random.choice(probs.shape[1], size= 1, p= probs[0,:])
            idxs = np.append(idxs, id_next, axis= 0)
        return idxs
