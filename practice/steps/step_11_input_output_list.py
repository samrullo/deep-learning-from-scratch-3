import numpy as np
from practice.steps.step_10_unittest import Variable, as_array


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, ys_grads):
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)


if __name__ == "__main__":
    inputs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    outputs = f(inputs)
    print(outputs[0].data)
