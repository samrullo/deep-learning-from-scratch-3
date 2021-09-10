from practice.dezero import Variable
import numpy as np
import practice.dezero.functions as F
import matplotlib.pyplot as plt

np.random.seed(0)
x = Variable(np.random.randn(100, 1))
y = 5 + 2 * x + np.random.randn(100, 1)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    return F.matmul(x, W) + b


def mean_squared_error(t, y):
    diff = t - y
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(f"{i + 1} itreaiton -> loss : {loss}, W:  {W}, b : {b}")

plt.scatter(x.data,y.data,color='blue')
plt.plot(x.data,y_pred.data,color='red')
plt.show()