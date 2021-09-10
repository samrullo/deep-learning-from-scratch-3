"""
Here we introduce Define-By-Run
We consider the relationship between Variable and Function.
What is Variable to Function? From Function's perspective Variable is something that goes into Function and out of it.
How about Variable?
What is Function to Variable? From Variable's point of view, it is created by a Function, hence Function is its creator
"""
import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, creator_func):
        self.creator = creator_func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, x: Variable):
        y = self.forward(x.data)
        output = Variable(y)
        output.set_creator(self)
        self.input = x
        self.output = output
        return output

    def forward(self, x_data):
        raise NotImplementedError()

    def backward(self, dout):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x_data):
        return x_data ** 2

    def backward(self, dout):
        x_data = self.input.data
        return dout * 2 * x_data


class Exp(Function):
    def forward(self, x_data):
        return np.exp(x_data)

    def backward(self, dout):
        x_data = self.input.data
        return dout * np.exp(x_data)


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    A = Square()
    B = Exp()
    C = Square()
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)

    # get the Function that created y
    C = y.creator

    # get C's input variable
    b = C.input

    # calculate b's grad
    b.grad = C.backward(y.grad)

    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)
    print(f"x_grad : {x.grad}")

    y.backward()
    print(f"x_grad using Variable backward method : {x.grad}")
