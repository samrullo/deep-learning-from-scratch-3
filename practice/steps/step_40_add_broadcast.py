from practice.dezero import Variable
import numpy as np

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 - x1
y.backward()
print(f"x0.grad : {x0.grad}")
print(f"x1.grad : {x1.grad}")