import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, x):
        y = self.forward(x.data)
        output = Variable(y)
        self.input = x
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
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(f"x.grad : {x.grad}")
