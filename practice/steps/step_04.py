from practice.steps.step_01 import Variable
import numpy as np
from practice.steps.step_02 import Square, Function
import math


def numerical_diff(f, x, eps):
    y0 = f(Variable(x.data - eps))
    y1 = f(Variable(x.data + eps))
    return (y1.data - y0.data) / (2 * eps)


if __name__ == "__main__":
    class Exp(Function):
        def forward(self, x):
            return np.exp(x)


    class TrigonSinus(Function):
        def forward(self, x):
            return np.sin(x.data)


    x = Variable(np.array(180 * math.pi / 180))
    func = TrigonSinus()
    grad = numerical_diff(func, x, 1e-4)
    print(grad)


    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))


    grad = numerical_diff(f, Variable(np.array(0.5)), 1e-4)
    print(f"composite function grad : {grad}")
