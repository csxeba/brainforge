import abc

import numpy as np
from .. import atomic, config
from ..util import zX_like, white_like


class LayerBase(abc.ABC):

    trainable = True

    def __init__(self, activation="linear", **kw):

        self.brain = None
        self.inputs = None
        self.output = None
        self.inshape = None
        self.op = None
        self.opb = None

        self.weights = None
        self.biases = None
        self.nabla_w = None
        self.nabla_b = None

        self.compiled = kw.get("compiled", config.compiled)
        self.trainable = kw.get("trainable", self.trainable)

        self.activation = atomic.activations[activation]() \
            if isinstance(activation, str) else activation

    def connect(self, to, inshape):
        self.brain = to
        self.inshape = inshape

    def shuffle(self) -> None:
        self.weights = white_like(self.weights)
        self.biases = zX_like(self.biases)

    def get_weights(self, unfold=True):
        if unfold:
            return np.concatenate((self.weights.ravel(), self.biases.ravel()))
        return [self.weights, self.biases]

    def set_weights(self, w, fold=True):
        if fold:
            W = self.weights
            self.weights = w[:W.size].reshape(W.shape)
            self.biases = w[W.size:].reshape(self.biases.shape)
        else:
            self.weights, self.biases = w

    @property
    def gradients(self):
        return np.concatenate([self.nabla_w.ravel(), self.nabla_b.ravel()])

    @property
    def nparams(self):
        return self.weights.size + self.biases.size

    @abc.abstractmethod
    def feedforward(self, X): raise NotImplementedError

    @abc.abstractmethod
    def backpropagate(self, delta): raise NotImplementedError

    @property
    @abc.abstractmethod
    def outshape(self): raise NotImplementedError

    @abc.abstractmethod
    def __str__(self): raise NotImplementedError


class NoParamMixin(abc.ABC):

    trainable = False

    def shuffle(self): pass

    def get_weights(self, unfold=True): pass

    def set_weights(self, w, fold=True): pass

    def gradients(self): pass

    @property
    def nparams(self):
        return None


class FFBase(LayerBase):

    """Base class for the fully connected layer types"""

    def __init__(self, neurons, activation="linear", **kw):
        LayerBase.__init__(self, activation, **kw)
        if not isinstance(neurons, int):
            neurons = np.prod(neurons)
        self.neurons = int(neurons)

    @abc.abstractmethod
    def connect(self, to, inshape):
        LayerBase.connect(self, to, inshape)
        self.nabla_w = zX_like(self.weights)
        self.nabla_b = zX_like(self.biases)

    @property
    def outshape(self):
        return self.neurons,
