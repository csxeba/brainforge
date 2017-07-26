import numpy as np

from ..util.typing import scalX, zX_like
from ..config import floatX

s0 = scalX(0.)
s05 = scalX(0.5)
s1 = scalX(1.)
s2 = scalX(2.)


class ActivationFunction:

    type = ""

    def forward(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __str__(self):
        return self.type

    def backward(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(ActivationFunction):

    type = "sigmoid"

    def forward(self, Z: np.ndarray):
        return s1 / (s1 + np.exp(-Z))
    
    def backward(self, A: np.ndarray) -> np.ndarray:
        return A * (s1 - A)


class HardSigmoid(ActivationFunction):

    type = "hardsigmoid"

    def forward(self, Z: np.ndarray):
        return np.clip((Z+s1)*s05, s0, s1)

    def backward(self, A: np.ndarray):
        mask = zX_like(A)
        mask[np.logical_and(A > 0., A < 1.)] = s05
        return A * mask


class Tanh(ActivationFunction):

    type = "tanh"

    def forward(self, Z) -> np.ndarray:
        return np.tanh(Z)
    
    def backward(self, A: np.ndarray) -> np.ndarray:
        return s1 - A**2


class Sqrt(ActivationFunction):

    type = "sqrt"

    def forward(self, Z) -> np.ndarray:
        return np.sqrt(Z)
    
    def backward(self, A: np.ndarray) -> np.ndarray:
        return s1 / (s2*A)


class Linear(ActivationFunction):

    type = "linear"

    def forward(self, Z) -> np.ndarray:
        return Z
    
    def backward(self, Z) -> np.ndarray:
        return s1


class ReLU(ActivationFunction):

    type = "relu"

    def forward(self, Z) -> np.ndarray:
        return np.maximum(s0, Z)
    
    def backward(self, A) -> np.ndarray:
        J = np.ones_like(A, dtype=floatX)
        J[A <= 0.] = s0
        return J


class SoftMax(ActivationFunction):

    type = "softmax"

    def __init__(self, temperature=1.):
        if temperature != 1.:
            self.temperature = scalX(temperature)
            self.__call__ = self.tn

    def tn(self, Z):
        return self.t1(Z / self.temperature)

    @staticmethod
    def t1(Z) -> np.ndarray:
        eZ = np.exp(Z - Z.max(axis=1, keepdims=True))
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    forward = t1

    def backward(self, A: np.ndarray) -> np.ndarray:
        return s1

    @staticmethod
    def true_backward(A: np.ndarray):
        # TODO: test this with numerical gradient testing!
        I = np.eye(A.shape[1], dtype=floatX)
        idx, idy = np.diag_indices(I)
        return A * (A[..., None] - I[None, ...])[:, idx, idy]


class OnePlus(ActivationFunction):

    type = "oneplus"

    def forward(self, Z):
        return 1. + np.log(1. + np.exp(Z))
    
    def backward(self, Z: np.ndarray):
        eZ = np.exp(Z)
        return eZ / (eZ + 1.)


activations = {"sigmoid": Sigmoid, "tanh": Tanh, "sqrt": Sqrt,
               "linear": Linear, "relu": ReLU, "softmax": SoftMax,
               "hard_sigmoid": HardSigmoid}
