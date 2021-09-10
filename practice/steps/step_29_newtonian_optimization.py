from practice.dezero import Variable
import numpy as np


def f(x):
    return x ** 4 - 2 * x ** 2


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(-2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward()
    x.data -= x.grad / gx2(x.data)

print(f"x after newtonian optimization of {iters} iterations : {x}")