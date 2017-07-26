import numpy as np
import numba as nb

from ._llutil import Xd
from ._llactivation import relu, tanh, sigmoid, sigmoid_p


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
        if t:
            E[t-1] += deltaZ[:, indim:]

    dX = E[:, :, :indim]
    nablab = np.sum(E, axis=(0, 1))
    return np.concatenate((dX.ravel(), nablaW.ravel(), nablab.ravel()))


@nb.jit(nopython=True)
def lstm_forward_tanh(X, W, b):
    outdim = W.shape[-1] // 4
    time, batch, indim = X.shape

    Z = np.zeros((time, batch, indim + outdim))
    O, C, Ca, cand, f, i, o = (np.zeros((time, batch, outdim)),
                               np.zeros((time, batch, outdim)),
                               np.zeros((time, batch, outdim)),
                               np.zeros((time, batch, outdim)),
                               np.zeros((time, batch, outdim)),
                               np.zeros((time, batch, outdim)),
                               np.zeros((time, batch, outdim)))

    for t in range(time):
        Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)

        p = np.dot(Z[t], W) + b
        p[:, :outdim * 3] = sigmoid(p[:, :outdim * 3])
        p[:, outdim * 3:] = tanh(p[:, outdim * 3:])

        f[t] = p[:, :outdim]
        i[t] = p[:, outdim:2 * outdim]
        o[t] = p[:, 2 * outdim:3 * outdim]
        cand[t] = p[:, 3 * outdim:]

        C[t] = C[t - 1] * f[t] + cand[t] * i[t]
        Ca[t] = tanh(C[t])

        O[t] = Ca[t] * o[t]

    cache = np.concatenate((C, Ca, cand, f, i, o), axis=-1)
    return np.concatenate((O.ravel(), Z.ravel(), cache.ravel()))


@nb.jit(nopython=True)
def lstm_backward(Z, bwO, E, W, cache, bwcache):
    do = W.shape[-1] // 4
    time, batch, zdim = Z.shape
    indim = zdim - do

    C, Ca, cand, f, i, o = (
        cache[:do], cache[do:2*do], cache[2*do:3*do],
        cache[3*do:4*do], cache[4*do:5*do], cache[5*do:]
    )

    nablaW = np.zeros_like(W)
    nablab = np.zeros(do * 4)

    delta = np.zeros_like(bwO[-1])
    deltaC = np.zeros_like(bwO[-1])
    deltaX = np.zeros((time, batch, indim))
    dgates = np.zeros((time, batch, do * 4))

    for t in range(time - 1, -1, -1):
        E[t] += delta
        deltaC += E[t] * o[t] * bwcache[t, :do]  # backwards Ca
        state_yesterday = 0. if not t else C[t - 1]
        df = state_yesterday * deltaC
        di = cand[t] * deltaC
        do = Ca[t] * E[t]
        dcand = i[t] * deltaC

        dgates[t] = np.concatenate((dcand, df, di, do), axis=-1) * bwcache[t, do:]  # backwards gates

        deltaC *= f[t]

        nablaW += np.dot(Z[t].T, dgates[t])
        nablab += np.sum(dgates[t], axis=0)

        deltaZ = np.dot(dgates[t], W.T)
        deltaX[t] = deltaZ[:, :-do]

    return deltaX, nablaW, nablab
