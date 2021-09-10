if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from practice.dezero import Variable
from practice.dezero.utils import plot_dot_graph
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s %(lineno)s]')


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
            30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


x = Variable(np.array(1))
y = Variable(np.array(1))
x.name = "x"
y.name = "y"
z = goldstein(x, y)
z.backward()
z.name = "output"
print(f"goldstein result={z}, x.grad, y.grad => {x.grad},{y.grad}")
plot_dot_graph(z, True, "goldstein_graph.png")
