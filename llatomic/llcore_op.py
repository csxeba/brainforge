from ._llops import dense_forward
from brainforge.util.typing import zX


class DenseOp:

    @staticmethod
    def forward(X, W, b=None):
        if b is None:
            b = zX(W.shape[-1])
        return dense_forward(X, W, b)

    @staticmethod
    def backward(X, E, W):
        raise NotImplementedError("No backwards for DenseOp!")
