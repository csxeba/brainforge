import abc
import warnings

import numpy as np


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
        eta = self.eta / m
        return W - gW * eta

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
        self.velocity = self.decay_velocity * self.velocity + (1 - self.decay_velocity) * gW
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

    def __init__(self, population, evolution_epochs=1,
                 survival_rate=0.3, mutation_rate=0.1,
                 optimize_accuracy=False,
                 *fitness_args, **fitness_kwargs):

        super().__init__()
        self.population = population
        self.optimize_accuracy = optimize_accuracy
        self.run_params = dict(epochs=evolution_epochs,
                               survival_rate=survival_rate,
                               mutation_rate=mutation_rate)
        self.fitness_args = fitness_args
        self.fitness_kwargs = fitness_kwargs

    def optimize(self, W, *args):
        self.population.run(force_update_at_every=3, verbosity=0,
                            *self.fitness_args, **self.fitness_kwargs,
                            **self.run_params)
        return self.population.best * 10

    def capsule(self, nosave=()):
        caps = {"optimize_accuracy": self.optimize_accuracy}
        caps.update(self.population.capsule())
        caps.update(self.run_params)
        return caps

    def __str__(self):
        return "Evolution"

    @staticmethod
    def default_fitness(ind, net, x, y, opt_acc):
        net.set_weights(ind)
        result = net.evaluate(x, y, classify=opt_acc)
        return (1. - result[-1]) if opt_acc else result


optimizers = {key.lower(): cls for key, cls in locals().items()
              if key.lower() not in ("np", "abc", "warnings", "evolution")}
