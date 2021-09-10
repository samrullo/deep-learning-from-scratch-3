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
            inputs, outputs = f.inputs, f.outputs
            gys = [out.grad for out in outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(inputs, gxs):
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
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


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    y.backward()
    print(f"x0.grad : {x0.grad}, x1.grad : {x1.grad}")
