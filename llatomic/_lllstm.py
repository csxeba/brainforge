import numba as nb
import numpy as np
from ._llactivation import sigmoid, tanh, relu


@nb.jit(nopython=True, cache=True)
def _lstm_update_state(p, T, outdim):
    # C[0], Ca[1], cand[2], f[3], i[4], o[5]
    p[:, outdim:] = sigmoid(p[:, outdim:])  # sigmoid to gates
    T[2] = p[:, :outdim]  # candidate
    T[3] = p[:, outdim:2*outdim]  # forget
    T[4] = p[:, 2*outdim:3*outdim]  # input
    T[5] = p[:, 3*outdim:]  # output
    T[0] = T[0] * T[3] + T[2] * T[4]  # Ct = Ct-1 * f + cand * i
    return T


@nb.jit(nopython=True, cache=True)
def lstm_forward_tanh(X, W, b):
    outdim = W.shape[-1] // 4
    time, batch, indim = X.shape

    Z = np.zeros((time, batch, indim + outdim))
    # C[0], Ca[1], cand[2], f[3], i[4], o[5]
    O = np.zeros((time, batch, outdim))
    T = np.zeros((time, 6, batch, outdim))

    for t in range(time):
        Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)
        p = np.dot(Z[t], W) + b
        p[:, :outdim] = tanh(p[:, :outdim])  # nonlin to candidate
        T[t] = _lstm_update_state(p, T[t], outdim)
        T[t, 1] = tanh(T[t, 0])  # nonlin to state
        O[t] = T[t, 1] * T[t, 5]  # O = f(C) * o
    return np.concatenate((O.ravel(), Z.ravel(), T.ravel()))


@nb.jit(nopython=True)
def lstm_forward_relu(X, W, b):
    outdim = W.shape[-1] // 4
    time, batch, indim = X.shape

    Z = np.zeros((time, batch, indim + outdim))
    # C[0], Ca[1], cand[2], f[3], i[4], o[5]
    O = np.zeros((time, batch, outdim))
    T = np.zeros((time, 6, batch, outdim))

    for t in range(time):
        Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)
        p = np.dot(Z[t], W) + b
        p[:, :outdim] = relu(p[:, :outdim])  # nonlin to candidate
        T[t] = _lstm_update_state(p, T[t], outdim)
        T[t, 1] = relu(T[t, 0])  # nonlin to state
        O[t] = T[t, 1] * T[t, 5]  # O = f(C) * o
    return np.concatenate((O.ravel(), Z.ravel(), T.ravel()))


@nb.jit(nopython=True)
def lstm_backward(Z, bwO, E, W, cache, bwcache):
    dimo = W.shape[-1] // 4
    time, batch, zdim = Z.shape
    indim = zdim - dimo

    C, Ca, cand, f, i, o = cache[0], cache[1], cache[2], cache[3], cache[4], cache[5]

    nablaW = np.zeros_like(W)
    nablab = np.zeros(dimo*4)
    deltaC = np.zeros_like(bwO[-1])
    deltaX = np.zeros((time, batch, indim))

    for t in range(time - 1, -1, -1):
        deltaC += E[t] * o[t] * bwcache[t, :, :dimo]  # backwards Ca
        state_yesterday = C[t-1]
        if not t:
            state_yesterday *= 0.

        do = Ca[t] * E[t]
        df = deltaC * state_yesterday
        di = deltaC * cand[t]
        dcand = deltaC * i[t]

        dgates = np.concatenate((dcand, df, di, do), axis=-1) * bwcache[t, :, dimo:]  # backwards gates
        deltaC *= f[t]

        nablaW += np.dot(Z[t].T, dgates)
        for ix in range(len(nablab)):
            nablab[ix] += dgates[:, ix].sum()
        deltaZ = np.dot(dgates, W.T)
        deltaX[t] = deltaZ[:, :-dimo]
        delta = deltaZ[:, indim:]
        if not t:
            delta *= 0.
        E[t-1] += delta

    return np.concatenate((deltaX.ravel(), nablaW.ravel(), nablab.ravel()))
