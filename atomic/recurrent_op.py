import numpy as np

from .activation_op import activations


class RecurrentOp:

    def __init__(self, activation):
        self.actfn = activations[activation]()

    def forward(self, X, W, b):
        outdim = W.shape[-1]
        time, batch, indim = X.shape
        O = np.zeros((time, batch, outdim))
        Z = np.zeros((time, batch, indim+outdim))
        for t in range(time):
            Z[t] = np.concatenate((X[t], O[t-1]), axis=-1)
            O[t] = self.actfn.forward(np.dot(Z[t], W) + b)
        return O, Z

    def backward(self, Z, O, E, W):
        outdim = W.shape[-1]
        time, batch, zdim = Z.shape
        indim = zdim - outdim
        bwO = self.actfn.backward(O)

        nablaW = np.zeros_like(W)
        nablab = np.zeros(outdim)

        delta = np.zeros_like(E[-1])
        deltaX = np.zeros((time, batch, indim))

        for t in range(time-1, -1, -1):
            delta += E[t]
            delta *= bwO[t]

            nablaW += np.dot(Z[t].T, delta)
            nablab += delta.sum(axis=0)

            deltaZ = np.dot(delta, W.T)
            deltaX[t] = deltaZ[:, :-outdim]
            delta = deltaZ[:, -outdim:]

        return deltaX, nablaW, nablab
