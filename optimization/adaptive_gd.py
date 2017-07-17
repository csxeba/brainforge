import numpy as np
from optimization.gradient_descent import SGD as _SGD


class Adagrad(_SGD):

    def __init__(self, nparams, eta=0.01, epsilon=1e-8, memory=None):
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


class RMSprop(Adagrad):

    def __init__(self, nparams, eta=0.1, decay=0.9, epsilon=1e-8, memory=None):
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


class Adam(_SGD):

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
        self.velocity = self.decay_velocity * self.velocity + (1. - self.decay_velocity) * gW
        self.memory = (self.decay_memory * self.memory + (1. - self.decay_memory) * (gW ** 2))
        update = (eta * self.velocity) / np.sqrt(self.memory + self.epsilon)
        return W - update

    def __str__(self):
        return "Adam"
