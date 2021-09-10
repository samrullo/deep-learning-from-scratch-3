import practice.dezero.functions as F
import numpy as np
from practice.dezero import Variable
from practice.dezero.utils import plot_dot_graph
from tqdm import tqdm

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 5

for i in tqdm(range(iters)):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = f'gx{iters}'
plot_dot_graph(gx, verbose=True, to_file='tanh.png', show_image=True)
