"""
we want to make Variable behave like regular ndarray types
for instance we want to be able to write a code like a*b
where both a and b are Variables and that to work
we also want it to work when one of them is not a Variable
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


def as_variable(obj):
    """
    convert ndarray into Variable
    :param obj: ndarray or Variable
    :return: Variable
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:
    # to make __rmul__ have precedence over ndarray __mul__
    __array_priority = 200

    def __init__(self, data, name=None):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} type is not supported")
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + ' ' * 9)
        return 'variable(' + p + ')'

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

    def __mul__(self, other):
        """
        to override multiplication method
        :param other: Variable other
        :return: multiplication result of two Variable types
        """
        return Mul()(self, other)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(input) for input in inputs]
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
    x1 = as_array(x1)
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


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, *y_grads):
        gy, = y_grads
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    """
    Multiply two Variables and return result
    :param x0: Variable ndarray one
    :param x1: Variable ndarray two
    :return: Variable multiplication of the two
    """
    x1 = as_array(x1)
    return Mul()(x0, x1)


# to override multiplication operator
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add

if __name__ == "__main__":
    x0 = Variable(np.array([2, 2]))
    x1 = Variable(np.array([3, 3]))
    y = mul(x0, x1)
    print(f"y : {y}")
    print(f"x0*x1 also returns right result : {x0 * x1}")
    print(f"x0+x1 works too : {x0 + x1}")
    print(f"x0+np.array(3) works : {x0 + np.array(3)}")
    print(f"x0+3 works : {x0 + 3}")
    print(f"x0*3 works : {x0 * 3}")
    print(f"3+x0 works : {3 + x0}")
    print(f"3*x0 works : {3 * x0}")

    a = Variable(np.array(3))
    b = Variable(np.array(2))
    c = Variable(np.array(5))

    y = a * b + c
    y.backward()

    print(f"y=a*b+c : {y}")
    print(f"a.grad : {a.grad}")
    print(f"b.grad : {b.grad}")
    print(f"c.grad : {c.grad}")
