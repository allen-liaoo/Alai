import numpy as np
from tqdm import tqdm
from alai import layers, models, loss as los

def gen_data(batch_size):
    input_data = np.random.randint(0, 2, (batch_size, 2))
    target_data = np.bitwise_xor(input_data[:, 0], input_data[:, 1])
    return input_data, target_data

linear1 = layers.Linear(2, 10)
act1 = layers.activation.Sigmoid()
linear2 = layers.Linear(10, 1)
act2 = layers.activation.Sigmoid()

num_iters = 10000
learning_rate = 1e-3
decay_rate = 0.5
k = 0
losses = np.zeros((num_iters, 1))
for iter in tqdm(range(num_iters), desc='Training MLP'):
    mini_batch_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    mini_batch_y = np.array([0, 1, 1, 0])
    mini_batch_y = mini_batch_y[:, None]

    if (iter + 1) % 1000 == 0:
        learning_rate = decay_rate * learning_rate

    dl_dw1_batch = np.zeros((10, 2))
    dl_db1_batch = np.zeros((10, 1))
    dl_dw2_batch = np.zeros((1, 10))
    dl_db2_batch = np.zeros((1, 1))

    batch_size = mini_batch_x.shape[0]
    ll = np.zeros((batch_size, 1))
    for i in range(batch_size):
        x = mini_batch_x[[i], :]
        y = mini_batch_y[[i], :]

        # forward propagation
        h1 = linear1.forward(x)
        h2 = act1.forward(h1)
        h3 = linear2.forward(h2)
        h4 = act2.forward(h3)

        # loss computation (forward + backward)
        l, dl_dy = los.MeanSquaredError()(h4, y)
        ll[i] = l

        # backward propagation
        dl_dh3 = act2.backward(dl_dy)
        dl_dh2, dl_dw2, dl_db2 = linear2.backward(dl_dh3)
        dl_dh1 = act1.backward(dl_dh2)
        dl_dx, dl_dw1, dl_db1 = linear1.backward(dl_dh1)

        # accumulate gradients
        dl_dw1_batch += dl_dw1
        dl_db1_batch += dl_db1
        dl_dw2_batch += dl_dw2
        dl_db2_batch += dl_db2

    losses[iter] = np.mean(ll)
    k = k + 1
    if k > len(mini_batch_x) - 1:
        k = 0

    # accumulate gradients
    linear1.update_weights(dl_dw1_batch / batch_size, dl_db1_batch / batch_size, lr=learning_rate)
    linear2.update_weights(dl_dw2_batch / batch_size, dl_db2_batch / batch_size, lr=learning_rate)


# evaluate
eval_input, eval_target = gen_data(1000)
eval_target = eval_target[:, None]
h1 = linear1.forward(eval_input)
h2 = act1.forward(h1)
h3 = linear2.forward(h2)
h4 = act2.forward(h3)
y = (h4 > 0.5).astype(int)
accuracy = np.sum(np.isclose(eval_target, y)) / len(eval_target)

print(f'average output for expected=1: {h4[eval_target == 1].mean()}')
print(f'average output for expected=0: {h4[eval_target == 0].mean()}')
print(f'accuracy: {accuracy}')

X_pt = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_pt = np.array([0, 1, 1, 0])
for _x, _y in zip(X_pt, Y_pt):
    h1 = linear1.forward(_x)
    h2 = act1.forward(h1)
    h3 = linear2.forward(h2)
    h4 = act2.forward(h3)
    print('Input:\t', list(map(int, _x)))
    print('Pred:\t', int(h4[0] > 0.5), h4[0].astype(float))
    print('Ouput:\t', int(_y))
    print('######')