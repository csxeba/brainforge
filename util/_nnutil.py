import numpy as np


floatX = "float64"


def ctx1(*arrays):
    return np.concatenate(arrays, axis=1)


def scalX(scalar, dtype=floatX):
    return np.asscalar(np.array([scalar], dtype=dtype))


def zX(*dims, dtype=floatX):
    return np.zeros(dims, dtype=dtype)


def zX_like(array, dtype=floatX):
    return zX(*array.shape, dtype=dtype)


def _densewhite(fanin, fanout):
    return np.random.randn(fanin, fanout) * np.sqrt(2. / float(fanin + fanout))


def _convwhite(nf, fc, fy, fx):
    return np.random.randn(nf, fc, fy, fx) * np.sqrt(2. / float(nf*fy*fx + fc*fy*fx))


def batch_stream(*arrays, m, shuffle=True, infinite=True):
    N = arrays[0].shape[0]
    while 1:
        arg = np.arange(N)
        if shuffle:
            np.random.shuffle(arg)
        shuffled = tuple(map(lambda ary: ary[arg], arrays))
        for start in range(0, N, m):
            minibatch = tuple(map(lambda ary: ary[start:start+m], shuffled))
            yield minibatch
        if not infinite:
            break


def white(*dims, dtype=floatX) -> np.ndarray:
    """Returns a white noise tensor"""
    tensor = _densewhite(*dims) if len(dims) == 2 else _convwhite(*dims)
    return tensor.astype(dtype)


def white_like(array, dtype=floatX):
    return white(*array.shape, dtype=dtype)


def rtm(A):
    """Converts an ndarray to a 2d array (matrix) by keeping the first dimension as the rows
    and flattening all the other dimensions to columns"""
    if A.ndim == 2:
        return A
    A = np.atleast_2d(A)
    return A.reshape(A.shape[0], np.prod(A.shape[1:]))


def describe(network):
    name = "{}, the Artificial Neural Network.".format(network.name) \
        if network.name else "BrainForge Artificial Neural Network."
    sep = "----------"
    chain = "\n".join((
        sep, name, sep, "Age: " + str(network.age),
        network.layerstack.describe(),
        sep
    ))
    return chain
