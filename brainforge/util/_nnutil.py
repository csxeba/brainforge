from brainforge import backend as xp


floatX = "float64"


def _densewhite(fanin, fanout):
    return xp.random.randn(fanin, fanout) * xp.sqrt(2. / float(fanin + fanout))


def _convwhite(nf, fc, fy, fx):
    return xp.random.randn(nf, fc, fy, fx) * xp.sqrt(2. / float(nf*fy*fx + fc*fy*fx))


def batch_stream(*arrays, m, shuffle=True, infinite=True):
    N = arrays[0].shape[0]
    while 1:
        arg = xp.arange(N)
        if shuffle:
            xp.random.shuffle(arg)
        shuffled = tuple(map(lambda ary: ary[arg], arrays))
        for start in range(0, N, m):
            minibatch = tuple(map(lambda ary: ary[start:start+m], shuffled))
            yield minibatch
        if not infinite:
            break


def white(*dims, dtype=floatX) -> xp.ndarray:
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
    A = xp.atleast_2d(A)
    return A.reshape(A.shape[0], xp.prod(A.shape[1:]))


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
