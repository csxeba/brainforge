from brainforge import backend as xp

from .abstract_optimizer import GradientDescent


class SGD(GradientDescent):

    def optimize(self, W, gW, m):
        return W - gW * (self.eta / m)

    def __str__(self):
        return "SGD"


class Momentum(SGD):

    def __init__(self, eta=0.1, mu=0.9):
        super().__init__(eta)
        self.mu = mu
        self.velocity = None

    def initialize(self, nparams, velocity=None):
        self.velocity = xp.zeros((nparams,)) if velocity is None else velocity

    def optimize(self, W, gW, m):
        eta = self.eta / m
        self.velocity = self.velocity * self.mu + gW * eta
        return W - self.velocity

    def __str__(self):
        return "Momentum"


class Nesterov(Momentum):

    def __init__(self, eta=0.1, mu=0.9):
        super().__init__(eta, mu)
        self.memory = None

    def initialize(self, nparams, velocity=None, memory=None):
        super().initialize(nparams, velocity)
        self.memory = xp.zeros_like(self.velocity) if memory is None else memory

    def optimize(self, W, gW, m):
        nabla = gW * (self.eta / m)
        W_ = self.memory - self.velocity + nabla
        self.memory = W
        self.velocity *= self.mu
        self.velocity += nabla
        return W_

    def __str__(self):
        return "Nesterov"


gdopt = {"sgd": SGD, "momentum": Momentum, "nesterov": Nesterov}
