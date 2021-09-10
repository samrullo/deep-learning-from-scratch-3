"""
we want to make Variable behave like regular ndarray types
for instance we want to be able to write a code like a*b
where both a and b are Variables and that to work
we also want it to work when one of them is not a Variable
we implement following operators as well
__neg__(self) => -self
__sub__(self,other) => self - other
__rsub__(self,other) => other - self
__truediv__(self,other) => self/other
__rtruediv__(self,other) => other/self
__pow__(self,other) => self ** other
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


class Neg(Function):
    def forward(self, x0):
        return -x0

    def backward(self, *y_grads):
        gy, = y_grads
        return -gy


def neg(x0):
    return Neg()(x0)


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, *y_grads):
        gy, = y_grads
        return gy, -gy


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
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy / x1, gy * (-x0 / x1 ** 2)


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
        x0 = self.inputs[0].data
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
