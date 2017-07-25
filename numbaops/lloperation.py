from ._llops import dense_forward
from ..util.typing import scalX

s0 = scalX(0.)


class DenseOp:

    @staticmethod
    def forward(X, W, b=s0):
        return dense_forward(X, W, b)

    @staticmethod
    def backward(X, E, W):
        raise NotImplementedError("No backwards for DenseOp!")

