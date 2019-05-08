import numba as nb
import numpy as np
from ._llactivation import sigmoid, tanh, relu


@nb.jit(nopython=True)
def lstm_forward(X, W, b, activation):
    outdim = W.shape[-1] // 4
    time, batch, indim = X.shape

    Z = np.zeros((time, batch, indim+outdim))
    O = np.zeros((time, batch, outdim))
    T = np.zeros((time, 6, batch, outdim))  # C[0], Ca[1], cand[2], f[3], i[4], o[5]

    for t in range(time):
        Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)
        p = np.dot(Z[t], W) + b
        p[:, :outdim] = activation(p[:, :outdim])  # nonlin to candidate

        p[:, outdim:] = sigmoid(p[:, outdim:])  # sigmoid to gates
        T[t, 2] = p[:, :outdim]  # candidate
        T[t, 3] = p[:, outdim:2*outdim]  # forget
        T[t, 4] = p[:, 2*outdim:3*outdim]  # input
        T[t, 5] = p[:, 3*outdim:]  # output
        T[t, 0] = T[t-1, 0] * T[t, 3] + T[t, 2] * T[t, 4]  # Ct = Ct-1 * f + cand * i

        T[t, 1] = activation(T[t, 0])  # nonlin to state
        O[t] = T[t, 1] * T[t, 5]  # O = f(C) * o
    return np.concatenate((O.ravel(), Z.ravel(), T.ravel()))


@nb.jit(nopython=True)
def lstm_backward(Z, bwO, E, W, cache, bwcache):
    # bwcache: bwCa, bwcand, bwf, bwi, bwo
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
        E[t-1] += deltaZ[:, indim:]

    return np.concatenate((deltaX.ravel(), nablaW.ravel(), nablab.ravel()))
