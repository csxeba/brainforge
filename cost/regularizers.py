import abc

import numpy as np
from ..util.typing import scalX

s05 = scalX(0.5)
s1 = scalX(1.)
s2 = scalX(2.)


class Regularizer(abc.ABC):

    def __init__(self, layer, lmbd=0.1):
        self.lmbd = lmbd
        self.layer = layer

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, eta, m):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class L1Norm(Regularizer):

    def __call__(self):
        return self.lmbd * np.abs(self.layer.get_weights()).sum()

    def derivative(self, eta, m):
        return ((eta * self.lmbd) / m) * np.sign(self.layer.get_weights())

    def __str__(self):
        return "L1-{}".format(self.lmbd)


class L2Norm(Regularizer):

    def __call__(self):
        return s05 * self.lmbd * np.linalg.norm(self.layer.get_weights() ** s2)

    def derivative(self, eta, m):
        return (s1 - ((eta * self.lmbd) / m)) * self.layer.get_weights().sum()

    def __str__(self):
        return "L2-{}".format(self.lmbd)
