import unittest
from practice.steps.step_09_function_improvement import square, Variable,as_array
import numpy as np


def numerical_gradient(f, x, eps=1e-4):
    x0 = Variable(as_array(x.data - eps))
    x1 = Variable(as_array(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2))
        y = square(x)
        self.assertEqual(y.data, np.array(4.0))

    def test_backward(self):
        x = Variable(np.array(3))
        y = square(x)
        y.backward()
        self.assertEqual(x.grad, np.array(6))

    def test_gradient(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        x_num_grad = numerical_gradient(square, x, eps=1e-4)
        self.assertTrue(np.allclose(x.grad, x_num_grad))


if __name__ == "__main__":
    unittest.main()
