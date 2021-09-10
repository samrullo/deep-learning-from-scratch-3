"""
In this step we demonstrate how we can chain Function classes.
For example consider an example of computing (e^(x^2))^2
"""
import numpy as np
from practice.steps.step_02 import Function, Square
from practice.steps.step_01 import Variable


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s %(lineno)s]')
    A = Square()
    B = Exp()
    C = Square()
    my_var = Variable(np.array(0.5))
    a = A(my_var)
    b = B(a)
    c = C(b)
    logging.info(f"final result : {c.data}")
