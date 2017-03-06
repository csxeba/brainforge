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


class CostFunction(abc.ABC):

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


class _XentOnSigmoid(CostFunction):

    def __call__(self, a: np.ndarray, y: np.ndarray):
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def derivative(outputs, targets):
        return np.subtract(outputs, targets)


class _XentOnSoftmax(CostFunction):

    def __call__(self, a: np.ndarray, y: np.ndarray):
        return -np.sum(y * np.log(a))

    @staticmethod
    def derivative(outputs, targets):
        return np.subtract(outputs, targets)

    def __str__(self):
        return "Xent"


class _Xent(CostFunction):

    def __init__(self, outact="softmax"):
        outact = str(outact).lower()
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


class _Cost:
    @property
    def mse(self):
        return _MSE()

    @property
    def xent(self):
        return _Xent()

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        item = item.lower()
        d = {str(fn).lower(): fn for fn in (_MSE(), _Xent(), _Hinge())}
        if item not in d:
            raise IndexError("Requested cost function is unsupported!")
        return d[item]


mse = _MSE()
xent = _Xent()
cost_fns = _Cost()
regularizers = _Regularizers()
