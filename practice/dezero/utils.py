import os
import subprocess
import logging
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def _dot_var(v, verbose=False):
    name = v.name if v.name else ''
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += f'{v.data.shape} {v.data.dtype}'
    dot_var = f'{id(v)} [label="{name}", color=orange, style=filled]\n'
    return dot_var


def _dot_func(f):
    txt = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    for x in f.inputs:
        txt += f"{id(x)}->{id(f)}\n"
    for y in f.outputs:
        txt += f"{id(f)}->{id(y())}\n"
    return txt


def get_dot_graph(out, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(out.creator)

    txt = txt + _dot_var(out, verbose)
    while funcs:
        f = funcs.pop()

        for x in f.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
        txt += _dot_func(f)
    return 'digraph g{\n' + txt + '}'


def plot_dot_graph(out, verbose=True, to_file="graph.png", show_image=True):
    dot_graph = get_dot_graph(out, verbose)
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    dot_file = to_file.split('.')[0] + '.dot'
    extension = os.path.splitext(to_file)[1][1:]
    with open(os.path.join(tmp_dir, dot_file), 'w') as fh:
        fh.write(dot_graph)
    cmd = f"dot {os.path.join(tmp_dir, dot_file)} -T {extension} -o {os.path.join(tmp_dir, to_file)}"
    subprocess.run(cmd, shell=True)
    logging.info(f"saved to {os.path.join(tmp_dir, to_file)}")
    if show_image:
        img = Image.open(os.path.join(tmp_dir, to_file))
        plt.imshow(img)
    return os.path.join(tmp_dir, to_file)


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
