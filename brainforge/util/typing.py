import numpy as np

from ..config import floatX


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


def white(*dims, dtype=floatX) -> np.ndarray:
    """Returns a white noise tensor"""
    tensor = _densewhite(*dims) if len(dims) == 2 else _convwhite(*dims)
    return tensor.astype(dtype)


def white_like(array, dtype=floatX):
    return white(*array.shape, dtype=dtype)


def emptyX(*dims):
    return np.empty(dims, dtype=floatX)
