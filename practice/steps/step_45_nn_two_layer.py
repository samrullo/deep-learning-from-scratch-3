from practice.dezero import Variable
from practice.dezero.layers import Linear
import numpy as np
import practice.dezero.functions as F
from practice.dezero import Model,Layer

class TwoLayerNet(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.l1=Linear(hidden_size)
        self.l2=Linear(out_size)

    def forward(self,x):
        y=F.sigmoid(self.l1(x))
        y=self.l2(y)
        return y



np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
model=TwoLayerNet(H,O)

lr = 0.2
iters = 10000

losses = []

for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)
    losses.append(loss.data)

    model.cleargrad()
    loss.backward()

    for param in model.params():
        param.data -= lr * param.grad.data
    if i % 100 == 0:
        print(f"{i} : loss {loss}")

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.scatter(x, y_pred.data, color='red')
plt.show()
