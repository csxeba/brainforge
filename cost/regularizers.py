import abc

import numpy as np


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
        return 0.5 * self.lmbd * (self.layer.get_weights()**2.).sum()

    def derivative(self, eta, m):
        return (1. - ((eta * self.lmbd) / m)) * self.layer.get_weights().sum()

    def __str__(self):
        return "L2-{}".format(self.lmbd)


class _Regularizers:

    def __init__(self):
        self.dct = {key.lower(): val for key, val in globals().items()
                    if key[:3] in ("L1N", "L2N")}

    def __getitem__(self, item):
        if item not in self.dct:
            raise IndexError("No such regularizer: {}".format(item))
        return self.dct[item]
