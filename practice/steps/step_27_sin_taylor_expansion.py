from practice.dezero import Variable, Function
import numpy as np
from practice.dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x0):
        return np.sin(x0)

    def backward(self, *gys):
        gy, = gys
        x0 = self.inputs[0].data
        return gy * np.cos(x0)


def sin(x0):
    return Sin()(x0)


import math


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(10000):
        t = (-1) ** i * (x ** (2 * i + 1)) / math.factorial(2 * i + 1)
        y += t
        if abs(t.data) <= threshold:
            break
    return y


if __name__ == "__main__":
    x0 = Variable(np.array(np.pi * 53 / 180))
    y = sin(x0)
    y.backward()
    print(f"sin({x0})={y}, x.grad={x0.grad}")
    x0 = Variable(np.array(np.pi * 53 / 180))
    x0.name="x"
    y = my_sin(x0,threshold=1e-150)
    y.backward()
    y.name="output"
    print(f"my_sin({x0})={y}, x.grad={x0.grad}")
    plot_dot_graph(y, True, 'sin_taylor_expansion.png')
