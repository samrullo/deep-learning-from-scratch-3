import numpy as np


def as_array(data):
    if np.isscalar(data):
        return np.array(data)


class Variable:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} type is not supported")
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, creator_func):
        self.creator = creator_func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, x: Variable):
        y = self.forward(x.data)
        output = Variable(as_array(y))
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


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(f"x.grad is : {x.grad}")
