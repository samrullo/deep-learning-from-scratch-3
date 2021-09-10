import numpy as np
from practice.steps.step_01 import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = x ** 2
        output = Variable(y)
        return output


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s %(lineno)s]')
    my_var = Variable(np.array(7))
    my_func = Function()
    my_out = my_func(my_var)
    logging.info(f"my output of the function is : {my_out.data}")
