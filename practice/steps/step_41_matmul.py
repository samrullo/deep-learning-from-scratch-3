from practice.dezero import Variable
import numpy as np
import practice.dezero.functions as F

x=Variable(np.random.randn(3,2))
W=Variable(np.random.randn(2,2))
y=F.matmul(x,W)
print(f"y.shape : {y.shape}")
y.backward()
print(f"x : {x} x.grad : {x.grad}")
print(f"W : {W}, W.grad  :{W.grad}")