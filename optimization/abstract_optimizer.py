import abc


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    def capsule(self, nosave=()):
        nosave = ["self"] + list(nosave)
        return {k: v for k, v in self.__dict__ if k not in nosave}


class GradientDescent(Optimizer):

    def __init__(self, eta=0.01):
        self.eta = eta
        self.nparams = None

    def initialize(self, **kw):
        pass

    @abc.abstractmethod
    def optimize(self, W, gW, m):
        raise NotImplementedError
