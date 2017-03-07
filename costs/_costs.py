import abc

import numpy as np


class CostFunction(abc.ABC):

    def __init__(self, brain):
        self.brain = brain

    def __call__(self, outputs, targets): pass

    def __str__(self): return ""

    @staticmethod
    @abc.abstractmethod
    def derivative(outputs, targets):
        raise NotImplementedError


class _MSE(CostFunction):

    def __call__(self, outputs, targets):
        return 0.5 * np.linalg.norm(outputs - targets) ** 2

    @staticmethod
    def derivative(outputs, targets):
        return np.subtract(outputs, targets)

    def __str__(self):
        return "MSE"


class _Xent(CostFunction):

    def __init__(self, brain):
        super().__init__(brain)
        outact = str(self.brain.layers[-1].activation)
        if outact not in ("sigmoid", "softmax"):
            msg = "Supplied output activation function ({})\n".format(outact)
            msg += "is not supported with the Cross-Entropy cost function!\n"
            msg += "Please choose either sigmoid or softmax as the output activation!"
            raise RuntimeError(msg)
        if outact == "sigmoid":
            self.__call__ = self.call_on_sigmoid
        else:
            self.__call__ = self.call_on_softmax
        self.derivative = self.simplified_derivative

    def __call__(self, a: np.ndarray, y: np.ndarray):
        return _Xent.call_on_softmax(a, y)

    @staticmethod
    def call_on_sigmoid(a: np.ndarray, y: np.ndarray):
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def call_on_softmax(a, y):
        return -np.sum(y * np.log(a))

    @staticmethod
    def derivative(outputs, targets):
        return _Xent.simplified_derivative(outputs, targets)

    @staticmethod
    def simplified_derivative(outputs, targets):
        return np.subtract(outputs, targets)

    @staticmethod
    def ugly_derivative(outputs, targets):
        enum = targets - outputs
        denom = (outputs - 1.) * outputs
        return enum / denom

    def __str__(self):
        return "Xent"


class _Hinge(CostFunction):

    def __call__(self, outputs, targets):
        return np.sum(np.maximum(0., 1. - targets * outputs))

    def __str__(self):
        return "Hinge"

    @staticmethod
    def derivative(outputs, targets):
        """
        Using subderivatives,
        d/da = -y whenever output > 0
        """
        out = -targets
        out[outputs > 1.] = 0.
        return out


class _CostFunctions:

    def __init__(self):
        self.dct = {"xent": _Xent,
                    "hinge": _Hinge,
                    "mse": _MSE}

    def __getitem__(self, item):
        if item not in self.dct:
            raise RuntimeError("No such cost function:", item)
        return self.dct[item]

cost_fns = _CostFunctions()
