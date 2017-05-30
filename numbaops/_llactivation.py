import numpy as np
import numba as nb

from ._llutil import nbfloatX, scalX, floatX

s0 = scalX(0., floatX)
s1 = scalX(1., floatX)
s2 = scalX(2., floatX)

finfout = "{t}({t})".format(t=nbfloatX)


@nb.vectorize((finfout,), nopython=True)
def sigmoid(Z):
    return s1 / (s1 + np.exp(-Z))


@nb.vectorize((finfout,), nopython=True)
def sigmoid_p(A):
    return A * (s1 - A)


@nb.vectorize((finfout,), nopython=True)
def tanh(Z):
    return np.tanh(Z)


@nb.vectorize((finfout,), nopython=True)
def tanh_p(A):
    return s1 - A ** s2


@nb.vectorize((finfout,), nopython=True)
def sqrt(Z):
    return np.sqrt(Z)


@nb.vectorize((finfout,), nopython=True)
def sqrt_p(A):
    return s1 / (s2*A)


@nb.vectorize((finfout,), nopython=True)
def linear(Z):
    return Z


@nb.vectorize((finfout,), nopython=True)
def linear_p(A):
    return s1


@nb.vectorize((finfout,), nopython=True)
def relu(Z):
    return np.maximum(s0, Z)


@nb.vectorize((finfout,), nopython=True)
def relu_p(A):
    J = np.ones_like(A)
    J[A <= s0] = s0
    return J


@nb.jit("{t}[:,::1]({t}[:,::1])".format(t=nbfloatX), nopython=True)
def softmax(Z):
    eZ = np.exp(Z)
    return eZ / np.sum(eZ, axis=1, keepdims=True)


@nb.jit("{t}[:,::1]({t}[:,::1]".format(t=nbfloatX), nopython=True)
def softmax_p(A):
    I = np.eye(A.shape[1], dtype=floatX)
    idx = np.arange(A.shape[1])
    return A * (A[..., None] - I[None, ...])[:, idx, idx]

