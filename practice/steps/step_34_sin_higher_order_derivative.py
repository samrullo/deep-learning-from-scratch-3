from practice.dezero import Variable
import numpy as np
import matplotlib.pyplot as plt
import practice.dezero.functions as F

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)
logs = [y.data.flatten()]
labels = ['y=sin(x)', "y'", "y''", "y'''"]
for i in range(3):
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

for i, log in enumerate(logs):
    plt.plot(x.data.flatten(), log, label=labels[i])

plt.legend(loc="lower right")
plt.show()
