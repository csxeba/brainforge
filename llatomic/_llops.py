import numpy as np
import numba as nb

from ._llutil import Xd
from ._llactivation import relu, relu_p, tanh, tanh_p


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
        act = relu(preact.ravel())
        O[t] = act.reshape(*preact.shape)
    return O


@nb.jit(nopython=True)
def recurrent_backward_relu(X, O, E, W):
    outdim = E.shape[-1]
    time, batch, indim = X.shape

    nablaW = np.zeros_like(W)
    deltaZ = np.zeros((time, batch, outdim))
    deltaX = np.zeros_like(X)

    for t in range(-1, -time, -1):
        Z = np.concatenate((X[t], O[t]), axis=-1)
        deltaZ[t] = deltaZ[t+1] + np.dot(E[t], W.T) * relu_p(O[t].ravel()).reshape(*O[t].shape)
        nablaW += np.dot(Z.T, deltaZ[t])
        deltaX[t] = deltaZ[t, :, :indim]

    nablab = np.sum(deltaZ, axis=(0, 1))
    return np.concatenate((np.ravel(deltaX), np.ravel(nablaW), np.ravel(nablab)))


@nb.jit(nopython=True)
def recurrent_forward_tanh(X, W, b):
    outdim = W.shape[-1]
    time, batch, indim = X.shape
    O = np.zeros((time, batch, outdim))
    Z = np.zeros((time, batch, indim+outdim))
    for t in range(time):
        Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)
        preact = np.dot(Z[t], W) + b
        act = tanh(preact.ravel())
        O[t] = act.reshape(*preact.shape)
    return np.concatenate((O.ravel(), Z.ravel()))


@nb.jit(nopython=True)
def recurrent_backward_tanh(Z, O, E, W):
    outdim = W.shape[-1]
    time, batch, zdim = Z.shape
    indim = zdim - outdim
    bwO = tanh_p(O.ravel()).reshape(O.shape)

    nablaW = np.zeros_like(W)
    nablab = np.zeros(outdim)

    delta = np.zeros_like(E[-1])
    deltaX = np.zeros((time, batch, indim))

    for t in range(time - 1, -1, -1):
        delta += E[t]
        delta *= bwO[t]

        nablaW += np.dot(Z[t].T, delta)
        nablab += delta.sum(axis=0)

        deltaZ = np.dot(delta, W.T)
        deltaX[t] = deltaZ[:, :-outdim]
        delta = deltaZ[:, -outdim:]

    return deltaX, nablaW, nablab
