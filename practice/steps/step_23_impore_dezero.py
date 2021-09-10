if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(3))
y = (x + 3) ** 2
y.backward()
print(f"y=({x}+3)**2 => {y}")
print(f"x.grad={x.grad}")
