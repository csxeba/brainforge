"""Wrappers for vector-operations and other functions"""
import numpy as np

from ..util.typing import scalX

s0 = scalX(0.)


class DenseOp:

    @staticmethod
    def forward(X, W, b=s0):
        return np.dot(X, W) + b

    @staticmethod
    def backward(X, E, W):
        dW = X.T @ E
        dX = E @ W.T
        db = E.sum(axis=0)
        return dW, db, dX


class ReshapeOp:

    type = "Reshape"

    @staticmethod
    def forward(X: np.ndarray, outshape: tuple):
        return X.reshape(-1, *outshape)

    @staticmethod
    def backward(E, inshape):
        return ReshapeOp.forward(E, inshape)
