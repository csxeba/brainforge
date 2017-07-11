import abc
import warnings

import numpy as np

from ..util import scalX

s1 = scalX(1.)


def _rms(X: np.ndarray):
    return np.sqrt((X**2).mean())


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


class GradientOptimizer(Optimizer):

    def __init__(self, nparams):
        self.nparams = nparams

    @abc.abstractmethod
    def optimize(self, W, gW, m):
        raise NotImplementedError


class SGD(GradientOptimizer):

    def __init__(self, nparams, eta=0.01):
        super().__init__(nparams)
        self.eta = eta

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


class Adagrad(SGD):

    def __init__(self, nparams, eta=0.01, epsilon=1e-8,
                 memory=None):
        warnings.warn("Adagrad is untested and possibly faulty!")
        super().__init__(nparams, eta)
        self.epsilon = epsilon
        self.memory = np.zeros((nparams,)) if memory is None else memory

    def optimize(self, W, gW, m):
        nabla = gW / m
        self.memory += nabla**2
        updates = (self.eta / np.sqrt(self.memory + self.epsilon)) * nabla
        return W - updates

    def __str__(self):
        return "Adagrad"


class RMSprop(Adagrad):

    def __init__(self, nparams, eta=0.1, decay=0.9, epsilon=1e-8,
                 memory=None):
        super().__init__(nparams, eta, epsilon, memory)
        self.decay = decay

    def optimize(self, W, gW, m):
        nabla = gW / m
        self.memory *= self.decay
        self.memory += (1. - self.decay) * (nabla ** 2.)
        updates = (self.eta * nabla) / np.sqrt(self.memory + self.epsilon)
        return W - updates

    def __str__(self):
        return "RMSprop"


# class Adadelta(SGD):
#
#     def __init__(self, rho=0.99, epsilon=1e-8, *args):
#         warnings.warn("Adadelta is untested and possibly faulty!", RuntimeWarning)
#         super().__init__(eta=0.0)
#         self.rho = rho
#         self.epsilon = epsilon
#         if not args:
#             self.gmemory = None
#             self.umemory = None
#         else:
#             if len(args) != 2:
#                 msg = "Invalid number of params for Adadelta! Got this:\n"
#                 msg += str(args)
#                 raise RuntimeError(msg)
#             self.gmemory, self.umemory = args
#         self._opt_coroutine = None
#
#     def connect(self, brain):
#         super().connect(brain)
#         if self.gmemory is None:
#             self.gmemory = np.zeros((self.brain.nparams,))
#         if self.umemory is None:
#             self.umemory = np.zeros((self.brain.nparams,))
#
#     def _opt_coroutine(self, rho, epsilon, gmemory, umemory):
#         update = np.zeros_like(gmemory)
#         W, gW, m = yield update
#         gmemory = rho * gmemory + (1. - rho) * gW**2
#
#     def optimize(self, W, gW, m):
#         W = self.brain.get_weights(unfold=True)
#         gW = self.brain.gradients
#         self.gmemory = self.rho * self.gmemory + (1. - self.rho) * gW**2
#         self.brain.set_weights(W - (self.umemory / self.gmemory) * gW)
#         update = (_rms(self.umemory) / _rms(gW)) * gW
#         self.umemory = self.rho * self.umemory + (1. - self.rho) * update**2


class Adam(SGD):

    def __init__(self, nparams, eta=0.1, decay_memory=0.9, decay_velocity=0.999,
                 epsilon=1e-8, velocity=None, memory=None):
        super().__init__(nparams, eta)
        self.decay_velocity = decay_velocity
        self.decay_memory = decay_memory
        self.epsilon = epsilon
        self.velocity = np.zeros((self.nparams,)) if velocity is None else velocity
        self.memory = np.zeros((self.nparams,)) if memory is None else memory

    def optimize(self, W, gW, m):
        eta = self.eta / m
        self.velocity = self.decay_velocity * self.velocity + (s1 - self.decay_velocity) * gW
        self.memory = (self.decay_memory * self.memory + (1 - self.decay_memory) * (gW ** 2))
        update = (eta * self.velocity) / np.sqrt(self.memory + self.epsilon)
        return W - update

    def __str__(self):
        return "Adam"


class Evolution(Optimizer):

    """
    Wrapper for brainforge.evolution.Population.
    Coordinates the differential evolution of weight learning.
    """

    def __init__(self, nparams=0, population=None, optimize_accuracy=False):

        super().__init__()
        if population is None:
            from ..evolution import Population
            if not nparams:
                raise RuntimeError("Please supply the number of weights to be optimized!")
            population = Population(nparams, self.default_fitness, fitness_weights=[1], limit=50)

        self.population = population
        self.optimize_accuracy = optimize_accuracy

    def optimize(self, net, x, y, epochs=1, survival_rate=0.1, mutation_rate=0.1,
                 force_update_at_every=0, verbosity=0):
        self.population.run(epochs, survival_rate, mutation_rate,
                            force_update_at_every, verbosity,
                            net=net, x=x, y=y, opt_acc=self.optimize_accuracy)
        best = self.population.best * 10.
        grade = self.population.grades.min()
        return best, grade

    def capsule(self, nosave=()):
        caps = {"optimize_accuracy": self.optimize_accuracy}
        caps.update(self.population.capsule())
        return caps

    def __str__(self):
        return "Evolution"

    @staticmethod
    def default_fitness(ind, net, x, y, opt_acc):
        net.set_weights(ind)
        result = net.evaluate(x, y, classify=opt_acc)
        return (1. - result[-1]) if opt_acc else result


optimizers = {key.lower(): cls for key, cls in locals().items()
              if key.lower() not in ("np", "abc", "warnings", "scalx")}
