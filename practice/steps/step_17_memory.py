"""
here we introduce usage of weakref to avoid cyclic reference between
objects
"""
import weakref
import numpy as np


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

    def backward(self):
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


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        self.generation = max([x.generation for x in inputs])
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

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


from memory_profiler import profile


@profile
def square(x):
    return Square()(x)


if __name__ == "__main__":
    for i in range(2):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
        print(f"y.data : {y.data}")
