import numpy as np
from alai.models.bigram import BigramLanguageModel
import os
import pathlib

# Load text dataset
path = pathlib.Path(__file__).parent / 'data/ios_messages/processed/'

text = ''
for filename in os.listdir(path):
    if not filename.endswith('.txt'): continue
    with open(path / filename, "r") as f:
        text += f.read()

# Get all characters used in the text
chars = sorted(list(set(text)))
chars_len = len(chars)
print(f'chars_len {chars_len}')
print('characters:' + ''.join(chars))

# Generate mapping from chars to integers
stoi = { ch:i for i, ch in enumerate(chars) } # use index as mapped integer
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Build dataset (ndarray) from text
data = np.array(encode(text))

# Split dataset into training and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(f'training: {train_data.shape}, validation: {val_data.shape}')

# Allow splitting data into batches
batch_size = 16
block_size = 8 # max context length for predictions

def get_batch(data):
    rind = np.random.randint(0, len(data) - block_size, size= batch_size)
    x = np.stack([data[i:i+block_size] for i in rind])
    y = np.stack([data[i+1:i+block_size+1] for i in rind]) # target of x is the next word, so offset by 1
    return x, y


bm = BigramLanguageModel(vocab_size= chars_len)

# print(decode(bm.generate(np.zeros((1,), dtype= int), max_new_tokens= 100)))

for epoch in range(100):
    trainx, trainy = get_batch(train_data)
    logits, loss, dl_dlogits = bm.forward(trainx, trainy)
    print(loss)
    bm.backward(dl_dlogits, learning_rate= 1e-3, update_weights= True)
    # print('done')
# print(decode(bm.generate(np.zeros((1,), dtype= int), max_new_tokens= 100)))