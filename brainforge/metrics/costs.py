import numpy as np

from ..util.typing import scalX

s0 = scalX(0.)
s05 = scalX(0.5)
s1 = scalX(1.)
s2 = scalX(2.)


class CostFunction:

    def __call__(self, outputs, targets): pass

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def derivative(outputs, targets):
        return outputs - targets


class _MeanSquaredError(CostFunction):

    def __call__(self, outputs, targets):
        return s05 * np.linalg.norm(outputs - targets) ** s2

    @staticmethod
    def true_derivative(outputs, targets):
        return outputs - targets


class _CategoricalCrossEntropy(CostFunction):

    def __call__(self, outputs: np.ndarray, targets: np.ndarray):
        return -(targets * np.log(outputs)).sum()

    @staticmethod
    def true_derivative(outputs, targets):
        enum = targets - outputs
        denom = (outputs - s1) * outputs
        return enum / denom


class _BinaryCrossEntropy(CostFunction):

    def __call__(self, outputs: np.ndarray, targets: np.ndarray):
        return -(targets * np.log(outputs) + (s1 - targets) * np.log(s1 - outputs)).sum()

    @staticmethod
    def true_derivative(outputs, targets):
        raise NotImplementedError


class _Hinge(CostFunction):

    def __call__(self, outputs, targets):
        return (np.maximum(s0, s1 - targets * outputs)).sum()

    @staticmethod
    def derivative(outputs, targets):
        """
        Using subderivatives,
        d/da = -y whenever output > 0
        """
        out = -targets
        out[outputs > s1] = s0
        return out


mean_squared_error = _MeanSquaredError()
categorical_crossentropy = _CategoricalCrossEntropy()
binary_crossentropy = _BinaryCrossEntropy()
hinge = _Hinge()

mse = mean_squared_error
cxent = categorical_crossentropy
bxent = binary_crossentropy

_costs = {k: v for k, v in locals().items() if k[0] != "_" and k != "CostFunction"}


def get(cost_function):
    if isinstance(cost_function, CostFunction):
        return cost_function
    cost = _costs.get(cost_function)
    if cost is None:
        raise ValueError("No such cost function: {}".format(cost))
    return cost
