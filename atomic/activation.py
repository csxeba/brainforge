import numpy as np

from ..util import scalX, floatX

s0 = scalX(0.)
s1 = scalX(1.)
s2 = scalX(2.)


class ActivationFunction:

    type = ""

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __str__(self):
        return self.type

    def derivative(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(ActivationFunction):

    type = "sigmoid"

    def __call__(self, Z: np.ndarray):
        return s1 / (s1 + np.exp(-Z))

    def derivative(self, A: np.ndarray) -> np.ndarray:
        return A * (s1 - A)


class Tanh(ActivationFunction):

    type = "tanh"

    def __call__(self, Z) -> np.ndarray:
        return np.tanh(Z)

    def derivative(self, A: np.ndarray) -> np.ndarray:
        return s1 - A**2


class Sqrt(ActivationFunction):

    type = "sqrt"

    def __call__(self, Z) -> np.ndarray:
        return np.sqrt(Z)

    def derivative(self, A: np.ndarray) -> np.ndarray:
        return s1 / (s2*A)


class Linear(ActivationFunction):

    type = "linear"

    def __call__(self, Z) -> np.ndarray:
        return Z

    def derivative(self, Z) -> np.ndarray:
        return s1


class ReLU(ActivationFunction):

    type = "relu"

    def __call__(self, Z) -> np.ndarray:
        return np.maximum(s0, Z)

    def derivative(self, A) -> np.ndarray:
        d = np.ones_like(A)
        d[A <= s0] = s0
        return d


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

    __call__ = t1

    def derivative(self, A: np.ndarray) -> np.ndarray:
        return s1

    @staticmethod
    def true_derivative(A: np.ndarray):
        # TODO: test this with numerical gradient testing!
        I = np.eye(A.shape[1], dtype=floatX)
        idx, idy = np.diag_indices(I)
        return A * (A[..., None] - I[None, ...])[:, idx, idy]


class OnePlus(ActivationFunction):

    type = "oneplus"

    def __call__(self, Z):
        return 1. + np.log(1. + np.exp(Z))

    def derivative(self, Z: np.ndarray):
        eZ = np.exp(Z)
        return eZ / (eZ + 1.)


activations = {"sigmoid": Sigmoid, "tanh": Tanh, "sqrt": Sqrt,
               "linear": Linear, "relu": ReLU, "softmax": SoftMax}
