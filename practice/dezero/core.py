"""
in core.py we change Variable and Function
to support multi-level gradients
"""
import weakref
import numpy as np
import contextlib
import practice.dezero


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
        if data is not None:
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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return practice.dezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return practice.dezero.functions.transpose(self, axes)

    @property
    def T(self):
        return practice.dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return practice.dezero.functions.sum(self, axis, keepdims)

    def set_creator(self, creator_func):
        self.creator = creator_func
        self.generation = self.creator.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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
            with using_config('enable_backprop', create_graph):
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

class Parameter(Variable):
    pass

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
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

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, y_grads):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, *y_grads):
        gy, = y_grads
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = practice.dezero.functions.sum_to(gy, self.x0_shape)
            gx1 = practice.dezero.functions.sum_to(gy, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, *y_grads):
        gy, = y_grads
        x, = self.inputs
        return gy * 2 * x


def square(x):
    return Square()(x)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, *y_grads):
        gy, = y_grads
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = practice.dezero.functions.sum_to(gx0, x0.shape)
            gx1 = practice.dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    """
    Multiply two Variables and return result
    :param x0: Variable ndarray one
    :param x1: Variable ndarray two
    :return: Variable multiplication of the two
    """
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x0):
        return -x0

    def backward(self, *y_grads):
        gy, = y_grads
        return -1 * gy


def neg(x0):
    return Neg()(x0)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, *y_grads):
        gy, = y_grads
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = practice.dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = practice.dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, -1 * gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    """
    to correctly calculate subtraction
    :param x0:
    :param x1:
    :return:
    """
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, *y_grads):
        gy, = y_grads
        x0, x1 = self.inputs
        gx0, gx1 = gy / x1, gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = practice.dezero.functions.sum_to(gx0, x0.shape)
            gx1 = practice.dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x0):
        return x0 ** self.c

    def backward(self, *y_grads):
        gy, = y_grads
        x0, = self.inputs
        return gy * self.c * (x0 ** (self.c - 1))


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    # to override multiplication operator
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = div
    Variable.__pow__ = pow
