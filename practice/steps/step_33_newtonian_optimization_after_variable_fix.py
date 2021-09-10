from practice.dezero import Variable
import numpy as np


def f(x):
    return x ** 4 - 2 * x ** 2


x = Variable(np.array(-2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    x.data -= gx.data / gx2.data

print(f"x after newtonian optimization of {iters} iterations : {x}")
