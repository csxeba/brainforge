import numpy as np

from .abstract_optimizer import GradientDescent


class SGD(GradientDescent):

    def optimize(self, W, gW, m):
        return W - gW * (self.eta / m)

    def __str__(self):
        return "SGD"


class Momentum(SGD):

    def __init__(self, nparams, eta=0.1, mu=0.9, velocity=None):
        super().__init__(nparams, eta)
        self.mu = mu
        self.velocity = np.zeros((nparams,)) if velocity is None else velocity

    def optimize(self, W, gW, m):
        eta = self.eta / m
        self.velocity = self.velocity * self.mu + gW * eta
        return W - self.velocity

    def __str__(self):
        return "Momentum"


class Nesterov(Momentum):

    def __init__(self, nparams, eta=0.1, mu=0.9, velocity=None, memory=None):
        super().__init__(nparams, eta, mu, velocity)
        self.memory = np.zeros_like(self.velocity) if memory is None else memory

    def optimize(self, W, gW, m):
        nabla = gW * (self.eta / m)
        W_ = self.memory - self.velocity + nabla
        self.memory = W
        self.velocity *= self.mu
        self.velocity += nabla
        return W_

    def __str__(self):
        return "Nesterov"

gdopt = {k.lower(): v for k, v in locals().items()
         if k not in ("GradientDescent", "np")
         and k[0] != "_"}
