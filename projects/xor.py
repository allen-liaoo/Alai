import numpy as np
from tqdm import tqdm
from alai import layers, models, loss as los

class XOR(models.Model):
    def __init__(self):
        super().__init__(lossFn=los.MeanSquaredError())
        self.linear1 = layers.Linear(2, 5)
        self.act1 = layers.activation.Sigmoid()
        self.linear2 = layers.Linear(5, 1)
        self.act2 = layers.activation.Sigmoid()

    def forward(self, x):
        y1 = self.linear1.forward(x) # (B, 2)
        z1 = self.act1.forward(y1) # (B, 2)
        y2 = self.linear2.forward(z1) # (B, 1)
        z2 = self.act2.forward(y2) # (B, 1)
        return z2

    def backward(self, dl_dz2):
        dl_dy2 = self.act2.backward(dl_dz2)
        dl_dz1, _, _ = self.linear2.backward(dl_dy2)
        dl_dy1 = self.act1.backward(dl_dz1)
        dl_dx, _, _ = self.linear1.backward(dl_dy1)

def gen_data(batch_size):
    X_pt = np.random.randint(0, 2, (batch_size, 2))
    Y_pt = np.bitwise_xor(X_pt[:, 0], X_pt[:, 1])[:, None]
    return X_pt, Y_pt

model = models.Sequential([
    layers.Linear(2, 10),
    layers.activation.Sigmoid(),
    layers.Linear(10, 1),
    layers.activation.Sigmoid()
], lossFn= los.BinaryCrossEntropy()) #.MeanSquaredError())
# model = XOR()
epoches = 100
print(model)

for i in tqdm(range(epoches), desc='Training XOR MLP'):
    # generate random binary data
    input_data, target_data = gen_data(15)

    y = model.forward(input_data)
    loss, dl_dy = model.compute_loss(y, target_data)
    model.backward(dl_dy, lr= 0.1)

# evaluate
X_pt, Y_pt = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])
for _x, _y in zip(X_pt, Y_pt):
    out = model.forward(_x)
    print('Input:\t', list(map(int, _x)))
    print('Pred:\t', int(out[0] > 0.5), out[0].astype(float))
    print('Ouput:\t', int(_y))
    print('######')

eval_input, eval_target = gen_data(1000)
y = model.forward(eval_input)
y_ = (y > 0.5).astype(int)

accuracy = np.sum(eval_target == y_) / len(eval_target)
print(f'average output for expected=1: {y[eval_target == 1].mean()}')
print(f'average output for expected=0: {y[eval_target == 0].mean()}')
print(f'accuracy: {accuracy}')