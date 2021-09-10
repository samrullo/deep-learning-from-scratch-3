from practice.dezero import Variable
import practice.dezero.functions as F
import numpy as np

x=Variable(np.array([[1,2,3],[4,5,6]]))
y=x.sum()
y.backward(retain_grad=True)
print(f"x.grad.shape : {x.grad.shape}")