import abc

import numpy as np

from ..util import scalX

s0 = scalX(0.)
s05 = scalX(0.5)
s1 = scalX(1.)
s2 = scalX(2.)


class CostFunction(abc.ABC):

    def __call__(self, outputs, targets): pass

    def __str__(self): return ""

    @staticmethod
    @abc.abstractmethod
    def derivative(outputs, targets):
        raise NotImplementedError


class MSE(CostFunction):

    def __call__(self, outputs, targets):
        return s05 * np.linalg.norm(outputs - targets) ** s2

    @staticmethod
    def derivative(outputs, targets):
        return outputs - targets

    def __str__(self):
        return "MSE"


class Xent(CostFunction):

    def __call__(self, outputs: np.ndarray, targets: np.ndarray):
        return Xent.call_on_softmax(outputs, targets)

    @staticmethod
    def call_on_sigmoid(outputs: np.ndarray, targets: np.ndarray):
        return -(targets * np.log(outputs) + (s1 - targets) * np.log(s1 - outputs)).sum()

    @staticmethod
    def call_on_softmax(outputs, targets):
        return -(targets * np.log(outputs)).sum()

    @staticmethod
    def derivative(outputs, targets):
        return Xent.simplified_derivative(outputs, targets)

    @staticmethod
    def simplified_derivative(outputs, targets):
        return outputs - targets

    @staticmethod
    def ugly_derivative(outputs, targets):
        enum = targets - outputs
        denom = (outputs - s1) * outputs
        return enum / denom

    def __str__(self):
        return "Xent"


class Hinge(CostFunction):

    def __call__(self, outputs, targets):
        return (np.maximum(s0, s1 - targets * outputs)).sum()

    def __str__(self):
        return "Hinge"

    @staticmethod
    def derivative(outputs, targets):
        """
        Using subderivatives,
        d/da = -y whenever output > 0
        """
        out = -targets
        out[outputs > s1] = s0
        return out


class _CostFunctions:

    def __init__(self):
        self.dct = {"xent": Xent,
                    "hinge": Hinge,
                    "mse": MSE}

    def __getitem__(self, item):
        if item not in self.dct:
            raise RuntimeError("No such cost function:", item)
        return self.dct[item]()

cost_functions = _CostFunctions()
