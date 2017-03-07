import abc
import warnings

import numpy as np


def _rms(X: np.ndarray):
    return np.sqrt((X**2).mean())


class Optimizer(abc.ABC):

    def __init__(self, eta=0.1):
        self.brain = None
        self.eta = eta

    def connect(self, brain):
        self.brain = brain

    @abc.abstractmethod
    def __call__(self, m):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def capsule(self):
        raise NotImplementedError


class SGD(Optimizer):

    def __call__(self, m):
        W = self.brain.get_weights(unfold=True)
        eta = self.eta / m
        self.brain.set_weights(W - self.brain.gradients * eta)

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

    def __call__(self, m):
        W = self.brain.get_weights(unfold=True)
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

    def __call__(self, m):
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

    def __call__(self, m):
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

    def __call__(self, m):
        W = self.brain.get_weights(unfold=True)
        gW = self.brain.gradients
        self.gmemory = self.rho * self.gmemory + (1. - self.rho) * gW**2
        self.brain.set_weights(W - (self.umemory / self.gmemory) * gW)
        update = (_rms(self.umemory) / _rms(gW)) * gW
        self.umemory = self.rho * self.umemory + (1. - self.rho) * update**2


class Adam(SGD):

    def __init__(self, eta=0.1, decay_memory=0.9, decay_velocity=0.999, epsilon=1e-8, *args):
        super().__init__(eta)
        self.decay_memory = decay_memory
        self.decay_velocity = decay_velocity
        self.epsilon = epsilon

        if not args:
            self.memory = None
            self.velocity = None
        else:
            if len(args) != 2:
                raise RuntimeError("Invalid number of params for ADAM! Got this:\n"
                                   + str(args))
            self.memory, self.velocity = args

    def connect(self, brain):
        super().connect(brain)
        if self.memory is None:
            self.memory = np.zeros((brain.nparams,))
        if self.velocity is None:
            self.velocity = np.zeros((brain.nparams,))

    def __call__(self, m):
        W = self.brain.get_weights(unfold=True)
        gW = self.brain.gradients
        eta = self.eta / m
        self.memory = self.decay_memory * self.memory + (1 - self.decay_memory) * gW
        self.velocity = (self.decay_velocity * self.velocity +
                         (1 - self.decay_velocity) * (gW ** 2))
        self.brain.set_weights(W - ((eta * self.memory) / (np.sqrt(self.velocity + self.epsilon))))

    def __str__(self):
        return "Adam"

    def capsule(self):
        param = [self.eta, self.decay_memory, self.decay_velocity, self.epsilon]
        # param += [self.mW, self.mb, self.vW, self.vb]
        return param


optimizers = {key.lower(): cls for key, cls in locals().items()
              if key not in ("np", "abc", "warnings")}
