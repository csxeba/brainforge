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

    def __init__(self, nparams, eta):
        self.eta = eta
        self.nparams = nparams

    @abc.abstractmethod
    def optimize(self, W, gW, m):
        raise NotImplementedError
