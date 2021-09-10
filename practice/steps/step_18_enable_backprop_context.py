"""
here we introduce technics to save memory
For instance in neural networks often we are only interested
in the gradients of parameters and input variables only
the gradients of intermediary variables can be thrown away
to make room in the memory
"""
import weakref
import numpy as np
import contextlib


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Config:
    enable_backprop = True


def as_array(data):
    if np.isscalar(data):
        return np.array(data)
    else:
        return data


class Variable:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} type is not supported")
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, creator_func):
        self.creator = creator_func
        self.generation = self.creator.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda _func: _func.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            inputs, outputs = f.inputs, f.outputs
            gys = [out().grad for out in outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            # if retain_grad is not True, throw away grads of all variables ahead of this one
            if not retain_grad:
                for output in outputs:
                    output().grad = None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]

        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *y_grads):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, *y_grads):
        gy, = y_grads
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, *y_grads):
        gy, = y_grads
        x, = self.inputs
        return gy * 2 * x.data


def square(x):
    return Square()(x)


if __name__ == "__main__":
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()
    print(f"x.grad : {x.grad[0:5, 0:5, 0:5]}")

    with no_grad():
        x = Variable(np.ones((100, 100, 100)))
        y = square(square(square(x)))
    print(f"y.data : {y.data[0:5, 0:5, 0:5]}")
