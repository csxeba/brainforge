import numpy as np

from .activation_op import activations
from ..util.typing import zX, zX_like

sigmoid = activations["sigmoid"]()


class RecurrentOp:

    def __init__(self, activation):
        self.actfn = activations[activation]()

    def forward(self, X, W, b):
        outdim = W.shape[-1]
        time, batch, indim = X.shape
        O = zX(time, batch, outdim)
        Z = zX(time, batch, indim+outdim)
        for t in range(time):
            Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)
            O[t] = self.actfn.forward(np.dot(Z[t], W) + b)
        return O, Z

    def backward(self, Z, O, E, W):
        outdim = W.shape[-1]
        time, batch, zdim = Z.shape
        indim = zdim - outdim
        bwO = self.actfn.backward(O)

        nablaW = zX_like(W)
        nablab = zX(outdim)

        delta = zX_like(E[-1])
        deltaX = zX(time, batch, indim)

        for t in range(time-1, -1, -1):
            delta += E[t]
            delta *= bwO[t]

            nablaW += np.dot(Z[t].T, delta)
            nablab += delta.sum(axis=0)

            deltaZ = np.dot(delta, W.T)
            deltaX[t] = deltaZ[:, :-outdim]
            delta = deltaZ[:, -outdim:]

        return deltaX, nablaW, nablab


class LSTMOp:

    def __init__(self, activation):
        self.actfn = activations[activation]()

    def forward(self, X, W, b):
        outdim = W.shape[-1] // 4
        time, batch, indim = X.shape

        Z = zX(time, batch, indim+outdim)
        O, C, f, i, o, cand, Ca = [zX(time, batch, outdim) for _ in range(7)]

        for t in range(time):
            Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)

            p = np.dot(Z[t], W) + b
            p[:, :outdim*3] = sigmoid.forward(p[:, :outdim*3])
            p[:, outdim*3:] = self.actfn.forward(p[:, outdim*3:])

            f[t] = p[:, :outdim]
            i[t] = p[:, outdim:2*outdim]
            o[t] = p[:, 2*outdim:3*outdim]
            cand[t] = p[:, 3*outdim:]

            C[t] = C[t] * f[t] + cand[t] * i[t]
            Ca[t] = self.actfn.forward(C[t])

            O[t] = Ca[t] * o[t]

        return O, (Z, C, f, i, o, cand, Ca)

    def backward(self, O, E, W, cache):
        Z, C, f, i, o, cand, Ca = cache
        outdim = W.shape[-1]
        time, batch, zdim = Z.shape
        indim = zdim - outdim

        bwgates = np.concatenate((f, i, o, cand), axis=-1)
        bwgates[:, :, :-outdim] = sigmoid.backward(bwgates[:, :, :-outdim])
        bwgates[:, :, -outdim:] = self.actfn.backward(bwgates[:, :, -outdim:])
        bwCa = self.actfn.backward(Ca)

        nablaW = zX_like(W)
        nablab = zX(outdim)

        delta = zX_like(O[-1])
        deltaC = zX_like(O[-1])
        deltaX = zX(time, batch, indim)

        for t in range(-1, -time, -1):
            E[t] += delta
            deltaC += E[t] * o * bwCa[t]
            state_yesterday = 0. if -t == time else C[t-1]
            df = state_yesterday * deltaC
            di = cand[t] * deltaC
            do = Ca[t] * E[t]
            dcand = i * deltaC

            dgates = np.concatenate((df, di, do, dcand), axis=-1) * bwgates[t]

            deltaC *= f[t]

            nablab += np.sum(dgates, axis=0)
            nablaW += np.dot(Z[t].T, dgates)

            deltaZ = np.dot(dgates, W.T)
            deltaX[t] = deltaZ[:, :-outdim]

        return deltaX, nablaW, nablab
