import abc

import numpy as np


def l1term(eta, lmbd, N):
    return (eta * lmbd) / N


def l2term(eta, lmbd, N):
    return 1 - ((eta * lmbd) / N)


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
