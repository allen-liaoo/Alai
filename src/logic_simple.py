import numpy as np
from alai import layers, models, loss as lo

class XOR(models.Model):
    def __init__(self):
        self.linear1 = layers.Linear(2, 2)
        self.sigmoid1 = layers.Sigmoid()
        self.linear2 = layers.Linear(2, 1)
        self.sigmoid2 = layers.Sigmoid()

    def forward(self, x, targets):
        x = self.linear1.forward(x) # (B, 2)
        x = self.sigmoid1.forward(x) # (B, 2)
        x = self.linear2.forward(x) # (B, 1)
        y = self.sigmoid2.forward(x) # (B, 1)
        targets = targets[:, None] # (B,) => (B, 1) for BCE

        loss, dl_dy = lo.BCE(y, targets)
        return y, loss, dl_dy

    def backward(self, dl_dy):
        dl_dy = self.sigmoid2.backward(dl_dy)
        dl_dx, _, _ = self.linear2.backward(dl_dy)
        dl_dy = self.sigmoid1.backward(dl_dx)
        dl_dx2, _, _ = self.linear1.backward(dl_dy)

model = XOR()
epoches = 100
for i in range(epoches):
    # generate random binary data
    input_data = np.random.randint(0, 2, (10, 2))
    target_data = np.bitwise_xor(input_data[:, 0], input_data[:, 1])

    y, loss, dl_dlogits = model.forward(input_data, target_data)
    model.backward(dl_dlogits)
    print(loss)

# evaluate
eval_input_data = np.random.randint(0, 2, (1000, 2))
eval_target_data = np.bitwise_xor(eval_input_data[:, 0], eval_input_data[:, 1])
y, _, _ = model.forward(eval_input_data, eval_target_data)
accuracy = np.sum(eval_target_data == np.argmax(y, axis= 1)) / len(eval_target_data)
print(f'accuracy: {accuracy}')