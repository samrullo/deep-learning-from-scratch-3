from practice.dezero import Variable
from practice.dezero.layers import Linear
import numpy as np
import practice.dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
l1 = Linear(H)
l2 = Linear(O)

lr = 0.2
iters = 10000


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

losses = []

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    losses.append(loss.data)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in (l1, l2):
        for param in l.params():
            param.data -= lr * param.grad.data
    if i % 100 == 0:
        print(f"{i} : loss {loss}")

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.scatter(x, y_pred.data, color='red')
plt.show()
