from ._llops import (
    recurrent_forward_relu,
    recurrent_forward_tanh, recurrent_backward
)
from .llactivation_op import act_fns


class RecurrentOp:

    def __init__(self, activation):
        if activation not in ("tanh", "relu"):
            raise RuntimeError("Only 'tanh' and 'relu' activations are supported here!")
        self.llact = act_fns[activation]
        self.bwlow = recurrent_backward
        self.fwlow = {"tanh": recurrent_forward_tanh,
                      "relu": recurrent_forward_relu}

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
