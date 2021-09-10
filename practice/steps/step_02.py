import numpy as np
from practice.steps.step_01 import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        """
        this should be implemented by the subclass
        :param x:
        :return:
        """
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s %(lineno)s]')
    my_var = Variable(np.array(7))
    my_func = Square()
    my_out = my_func(my_var)
    logging.info(f"my output of the function is : {my_out.data}")
