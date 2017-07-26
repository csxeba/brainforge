import numpy as np

from ._llops import (
    recurrent_forward_relu, recurrent_forward_tanh,
    recurrent_backward,
    lstm_forward_tanh,
    lstm_backward
)
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

    def __init__(self, activation):
        super().__init__(activation)
        self.fwlow = {
            "tanh": lstm_forward_tanh
        }[activation.lower()]

    def forward(self, X, W, b):
        do = W.shape[-1] // 4
        t, m, di = X.shape

        Oshape = (t, m, do)
        Zshape = (t, m, do+di)
        cacheshape = (t, m, do*6)
        Obord = np.prod(Oshape)
        Zbord = np.prod(Zshape) + Obord

        vector = self.fwlow(X, W, b)

        O = vector[:Obord].reshape(*Oshape)
        Z = vector[Obord:Zbord].reshape(*Zshape)
        cache = vector[Zbord:].reshape(*cacheshape)
        return O, Z, cache

    def backward(self, Z, O, E, W, cache):
        m, t, dz = Z.shape
        do = W.shape[-1] // 4
        di = dz - do
        g = m*t*di
        bwcache = cache[..., do:].copy()
        bwcache[..., :2*do] = self.llact.backward(bwcache[..., :2*do])
        bwcache[..., 2*do:] = sigmoid.backward(bwcache[..., 2*do:])
        vector = lstm_backward(Z, self.llact.backward(O),
                               E, W, cache, bwcache)
        dX = vector[:g].reshape(t, m, di)
        gW = vector[g:g+W.size].reshape(W.shape)
        gb = vector[g+W.size:]
        return dX, gW, gb
