import abc
import warnings

import numpy as np


def _rms(X: np.ndarray):
    return np.sqrt((X**2).mean())


class Optimizer(abc.ABC):

    def __init__(self):
        self.brain = None

    def connect(self, brain):
        self.brain = brain

    @abc.abstractmethod
    def optimize(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def capsule(self):
        raise NotImplementedError


class GradientOptimizer(Optimizer):

    @abc.abstractmethod
    def optimize(self, *args):
        self.brain.backpropagate()
        return self.brain.get_weights(unfold=True), self.brain.gradients


class SGD(GradientOptimizer):

    def __init__(self, eta):
        super().__init__()
        self.eta = eta

    def optimize(self, m, *args):
        W, gW = super().optimize(m)
        eta = self.eta / m
        self.brain.set_weights(W - gW * eta)

    def __str__(self):
        return "SGD"

    def capsule(self):
        return [self.eta]


class Momentum(SGD):

    def __init__(self, eta=0.1, mu=0.9, nesterov=False, *args):
        super().__init__(eta)
        self.mu = mu
        self.nesterov = nesterov
        if not args:
            self.velocity = None
        else:
            if len(args) != 1:
                msg = "Invalid number of params for Momentum! Got this:\n"
                msg += str(args)
                raise RuntimeError(msg)
            self.velocity = args[0]

    def connect(self, brain):
        super().connect(brain)
        if self.velocity is None:
            self.velocity = np.zeros((brain.nparams,))

    def optimize(self, m, *args):
        W, gW = self.brain.get_weights(unfold=True)
        eta = self.eta / m
        self.velocity *= self.mu
        deltaW = W + self.velocity if self.nesterov else self.brain.gradients
        self.velocity += deltaW * eta
        self.brain.set_weights(W - self.velocity)

    def __str__(self):
        return ("Nesterov " if self.nesterov else "") + "Momentum"

    def capsule(self):
        return [self.eta, self.mu, self.nesterov]  # , self.vW, self.vb]


class Adagrad(SGD):

    def __init__(self, eta=0.01, epsilon=1e-8, *args):
        warnings.warn("Adagrad is untested and possibly faulty!")
        super().__init__(eta)
        self.epsilon = epsilon
        if not args:
            self.memory = None
        else:
            if len(args) != 1:
                msg = "Invalid number of params for {}! Got this:\n".format(str(self))
                msg += str(args)
                raise RuntimeError(msg)
            self.memory = args[0]

    def connect(self, brain):
        super().connect(brain)
        if self.memory is None:
            self.memory = np.zeros((brain.nparams,))

    def optimize(self, m, *args):
        W = self.brain.get_weights(unfold=True)
        eta = self.eta / m
        self.memory += self.brain.gradients ** 2
        self.brain.set_weights(W - (eta / np.sqrt(self.memory + self.epsilon)) * self.brain.gradients)

    def __str__(self):
        return "Adagrad"

    def capsule(self):
        return [self.eta, self.epsilon]  # , self.mW, self.mb]


class RMSprop(Adagrad):

    def __init__(self, eta=0.1, decay=0.9, epsilon=1e-8, *args):
        super().__init__(eta, epsilon, *args)
        self.decay = decay

    def optimize(self, m, *args):
        W = self.brain.get_weights(unfold=True)
        gW = self.brain.gradients
        eta = self.eta / m
        self.memory = self.decay * self.memory + (1 - self.decay) * (gW ** 2)
        self.brain.set_weights(W - ((eta * gW) / (np.sqrt(self.memory + self.epsilon))))

    def __str__(self):
        return "RMSprop"

    def capsule(self):
        return [self.eta, self.decay, self.epsilon]  # , self.mW, self.mb]


class Adadelta(SGD):

    def __init__(self, rho=0.99, epsilon=1e-8, *args):
        warnings.warn("Adadelta is untested and possibly faulty!", RuntimeWarning)
        super().__init__(eta=0.0)
        self.rho = rho
        self.epsilon = epsilon
        if not args:
            self.gmemory = None
            self.umemory = None
        else:
            if len(args) != 2:
                msg = "Invalid number of params for Adadelta! Got this:\n"
                msg += str(args)
                raise RuntimeError(msg)
            self.gmemory, self.umemory = args

    def connect(self, brain):
        super().connect(brain)
        if self.gmemory is None:
            self.gmemory = np.zeros((self.brain.nparams,))
        if self.umemory is None:
            self.umemory = np.zeros((self.brain.nparams,))

    def optimize(self, m, *args):
        W = self.brain.get_weights(unfold=True)
        gW = self.brain.gradients
        self.gmemory = self.rho * self.gmemory + (1. - self.rho) * gW**2
        self.brain.set_weights(W - (self.umemory / self.gmemory) * gW)
        update = (_rms(self.umemory) / _rms(gW)) * gW
        self.umemory = self.rho * self.umemory + (1. - self.rho) * update**2


class Adam(SGD):

    def __init__(self, eta=0.1, decay_memory=0.9, decay_velocity=0.999, epsilon=1e-8, *args):
        super().__init__(eta)
        self.decay_velocity = decay_velocity
        self.decay_memory = decay_memory
        self.epsilon = epsilon

        if not args:
            self.velocity = None
            self.memory = None
        else:
            if len(args) != 2:
                raise RuntimeError("Invalid number of params for ADAM! Got this:\n"
                                   + str(args))
            self.velocity, self.memory = args

    def connect(self, brain):
        super().connect(brain)
        if self.velocity is None:
            self.velocity = np.zeros((brain.nparams,))
        if self.memory is None:
            self.memory = np.zeros((brain.nparams,))

    def optimize(self, m, *args):
        W = self.brain.get_weights(unfold=True)
        gW = self.brain.gradients
        eta = self.eta / m
        self.velocity = self.decay_velocity * self.velocity + (1 - self.decay_velocity) * gW
        self.memory = (self.decay_memory * self.memory +
                       (1 - self.decay_memory) * (gW ** 2))
        update = ((eta * self.velocity) / (np.sqrt(self.memory + self.epsilon)))
        self.brain.set_weights(W - update)

    def __str__(self):
        return "Adam"

    def capsule(self):
        param = [self.eta, self.decay_velocity, self.decay_memory, self.epsilon]
        # param += [self.mW, self.mb, self.vW, self.vb]
        return param


class Evolution(Optimizer):

    """
    Wrapper for brainforge.evolution.Population.
    Coordinates the differential evolution of weight learning.
    """

    def __init__(self, survrate=0.3, mutrate=0.1, limit=100, evol_epoch_per_batch=1,
                 optimize_accuracy=False, mate_function: callable=None,
                 grade_function: callable=None, mutate_function: callable=None):

        super().__init__()
        self.survrate = survrate
        self.mutrate = mutrate
        self.limit = limit
        self.optimize_accuracy = optimize_accuracy
        self.pop = None
        self.pop_params = {kw: fn for kw, fn in locals().items() if "_fn" == kw[-3:]}
        self.evolepochs = evol_epoch_per_batch

    @classmethod
    def from_population(cls, population, survrate=0.3, mutrate=0.1, limit=100,
                        evol_epoch_per_batch=1, optimize_accuracy=False):
        opti = cls(survrate, mutrate, limit, evol_epoch_per_batch, optimize_accuracy)
        opti.pop = population

    def connect(self, brain):
        from ..evolution import Population
        super().connect(brain)
        if self.pop is None:
            self.pop = Population(self.brain.nparams,
                                  self._fitness,
                                  fitness_weights=[1.],
                                  limit=self.limit,
                                  **self.pop_params)
        else:
            inds, params = self.pop.individuals.shape
            assert inds == self.limit and params == self.brain.nparams

    def optimize(self, *args):
        self.pop.run(self.evolepochs, self.survrate, self.mutrate,
                     force_update_at_every=3, verbosity=0)
        self.brain.set_weights(self.pop.best*10)

    def _fitness(self, gen):
        self.brain.set_weights(gen*10., fold=True)
        if self.optimize_accuracy:
            cost, acc = self.brain.evaluate(self.brain.X, self.brain.Y, classify=True)
            grade = 1. - acc
        else:
            grade = self.brain.evaluate(self.brain.X, self.brain.Y, classify=False)
        return grade

    def capsule(self):
        param = [self.survrate, self.mutrate, self.limit, self.evolepochs,
                 self.optimize_accuracy]
        pp = self.pop_params
        param += [pp["mate_function"], pp["grade_function"], pp["mutate_function"]]
        # param += [self.pop.individuals.ravel()]
        return param

    def __str__(self):
        return "Evolution"


optimizers = {key.lower(): cls for key, cls in locals().items()
              if key not in ("np", "abc", "warnings")}


