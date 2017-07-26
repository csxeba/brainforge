import numpy as np
import numba as nb

from ._llutil import Xd
from ._llactivation import relu, tanh


@nb.jit("{f2}({f2},{f2},{f1})".format(f1=Xd(1), f2=Xd(2)), nopython=True)
def dense_forward(X, W, b):
    return np.dot(X, W) + b


@nb.jit(nopython=True)
def recurrent_forward_relu(X, W, b):
    outdim = W.shape[-1]
    time, batch, indim = X.shape
    O = np.zeros((time, batch, outdim))
    for t in range(time):
        Z = np.concatenate((X[t], O[t-1]), axis=-1)
        preact = np.dot(Z, W) + b
        O[t] = relu(preact.ravel()).reshape(*preact.shape)
    return O


@nb.jit(nopython=True)
def recurrent_forward_tanh(X, W, b):
    outdim = W.shape[-1]
    time, batch, indim = X.shape
    O = np.zeros((time, batch, outdim))
    Z = np.zeros((time, batch, indim+outdim))
    for t in range(time):
        Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)
        preact = np.dot(Z[t], W) + b
        O[t] = tanh(preact.ravel()).reshape(*preact.shape)
    return np.concatenate((O.ravel(), Z.ravel()))


@nb.jit(nopython=True)
def recurrent_backward(Z, bwO, E, W):
    indim = Z.shape[-1] - bwO.shape[-1]
    nablaW = np.zeros_like(W)

    for t in range(Z.shape[0]-1, -1, -1):
        E[t] *= bwO[t]
        nablaW += np.dot(Z[t].T, E[t])
        deltaZ = np.dot(E[t], W.T)
        E[t - 1] += deltaZ[:, indim:]

    dX = E[:, :, :indim]
    nablab = np.sum(E, axis=(0, 1))
    return np.concatenate((dX.ravel(), nablaW.ravel(), nablab.ravel()))
