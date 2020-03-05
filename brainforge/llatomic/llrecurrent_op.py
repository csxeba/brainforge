import numpy as np

from ._llrecurrent import recurrent_forward_relu, recurrent_forward_tanh, recurrent_backward
from ._lllstm import lstm_forward, lstm_backward
from .llactivation_op import llactivations

sigmoid = llactivations["sigmoid"]()


class ROpBase:

    def __init__(self, activation):
        if activation not in ("tanh", "relu"):
            raise RuntimeError("Only 'tanh' and 'relu' activations are supported here!")
        self.llact = llactivations[activation]()
        self.bwlow = recurrent_backward
        self.fwlow = None


class RecurrentOp(ROpBase):

    def __init__(self, activation):
        super().__init__(activation)
        self.fwlow = {
            "tanh": recurrent_forward_tanh, "relu": recurrent_forward_relu
        }[activation.lower()]

    def forward(self, X, W, b):
        t, m, di = X.shape
        do, = b.shape
        g = t*m*do
        vector = self.fwlow(X, W, b)
        O = vector[:g].reshape(t, m, do)
        Z = vector[g:].reshape(t, m, di+do)
        return O, Z

    def backward(self, Z, O, E, W):
        t, m, z = Z.shape
        do = O.shape[-1]
        di = z - do
        g = t*m*di
        vector = self.bwlow(Z, self.llact.backward(O), E, W)
        dX = vector[:g].reshape(t, m, di)
        gW = vector[g:g+W.size].reshape(W.shape)
        gb = vector[g+W.size:]
        return dX, gW, gb


class LSTMOp(ROpBase):

    def forward(self, X, W, b):
        do = W.shape[-1] // 4
        t, m, di = X.shape

        Oshape = (t, m, do)
        Zshape = (t, m, do+di)
        cacheshape = (t, 6, m, do)
        Obord = np.prod(Oshape)
        Zbord = np.prod(Zshape) + Obord

        vector = lstm_forward(X, W, b, self.llact.llact)

        O = vector[:Obord].reshape(*Oshape)
        Z = vector[Obord:Zbord].reshape(*Zshape)
        cache = vector[Zbord:].reshape(*cacheshape)
        return O, Z, cache.transpose(1, 0, 2, 3)

    def backward(self, Z, O, E, W, cache):
        do = W.shape[-1] // 4
        m, t, dz = Z.shape
        di = dz - do
        g = m*t*di

        bwcache = cache[1:].copy()
        bwcache[:2] = self.llact.backward(bwcache[:2])
        bwcache[2:] = sigmoid.backward(bwcache[2:])
        bwO = self.llact.backward(O)
        vector = lstm_backward(Z, bwO, E, W, cache, np.concatenate(bwcache, axis=-1))
        dX = vector[:g].reshape(t, m, di)
        gW = vector[g:g+W.size].reshape(W.shape)
        gb = vector[g+W.size:]
        return dX, gW, gb
