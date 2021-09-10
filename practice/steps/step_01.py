import numpy as np

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s %(lineno)s]')


class Variable:
    def __init__(self, data):
        self.data = data


if __name__ == "__main__":
    my_var = Variable(np.array(2))
    logging.info(f"my variable data is {my_var.data}")
    my_var.data = np.array(7)
    logging.info(f"my variable data after changing its value : {my_var.data}")
