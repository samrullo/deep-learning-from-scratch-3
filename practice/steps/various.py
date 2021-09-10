from practice.dezero import Variable
import numpy as np

x=Variable(np.random.randn(2,3,4,5))
y=x.reshape(2,12,5)
print(f"y.shape : {y.shape}")

y=x.transpose(3,0,1,2)
print(f"transposede y.shape : {y.shape}")