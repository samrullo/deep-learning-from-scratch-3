import practice.dezero.functions as F
from practice.dezero import Variable
import numpy as np

x=Variable(np.array([[1,2,3],[4,5,6]]))
y=F.reshape(x,6)
y.backward(retain_grad=True)
print(f"y.grad.shape : {y.grad.shape}")
print(f"x.grad.shape : {x.grad.shape}")