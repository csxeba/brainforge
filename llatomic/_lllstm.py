import numba as nb
import numpy as np
from ._llactivation import sigmoid, tanh, relu


@nb.jit(nopython=True)
def _lstm_update_state(p, T, outdim):
    p[:, :outdim * 3] = sigmoid(p[:, :outdim * 3])
    T[4] = p[:, :outdim]
    T[5] = p[:, outdim:2 * outdim]
    T[6] = p[:, 2 * outdim:3 * outdim]
    T[3] = p[:, 3 * outdim:]
    T[1] = T[1] * T[4] + T[3] * T[5]
    return T


@nb.jit(nopython=True)
def lstm_forward_tanh(X, W, b):
    outdim = W.shape[-1] // 4
    time, batch, indim = X.shape

    Z = np.zeros((time, batch, indim + outdim))
    # O[0], C[1], Ca[2], cand[3], f[4], i[5], o[6]
    T = np.zeros((time, 7, batch, outdim))

    for t in range(time):

        Z[t] = np.concatenate((X[t], T[t-1, 0]), axis=-1)

        p = np.dot(Z[t], W) + b
        p[:, outdim*3:] = tanh(p[:, outdim*3:])

        T[t] = _lstm_update_state(p, T[t], outdim)

        T[t, 2] = tanh(T[t, 1])
        T[t, 0] = T[t, 2] * T[t, 6]

    cache = np.concatenate(T[1:], axis=-1)
    return np.concatenate((T[0].ravel(), Z.ravel(), cache.ravel()))


@nb.jit(nopython=True)
def lstm_forward_relu(X, W, b):
    outdim = W.shape[-1] // 4
    time, batch, indim = X.shape

    Z = np.zeros((time, batch, indim + outdim))
    # O[0], C[1], Ca[2], cand[3], f[4], i[5], o[6]
    T = np.zeros((time, 7, batch, outdim))

    for t in range(time):
        Z[t] = np.concatenate((X[t], T[t - 1, 0]), axis=-1)

        p = np.dot(Z[t], W) + b
        p[:, outdim * 3:] = relu(p[:, outdim * 3:])

        T[t] = _lstm_update_state(p, T[t], outdim)

        T[t, 2] = relu(T[t, 1])
        T[t, 0] = T[t, 2] * T[t, 6]

    cache = np.concatenate(T[1:], axis=-1)
    return np.concatenate((T[0].ravel(), Z.ravel(), cache.ravel()))


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
