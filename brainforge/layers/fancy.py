import numpy as np

from .abstract_layer import LayerBase, NoParamMixin, FFBase
from ..atomic import Sigmoid
from ..util import rtm, scalX, zX, white
from ..config import floatX

sigmoid = Sigmoid()


class Highway(FFBase):
    """
    Neural Highway Layer based on Srivastava et al., 2015
    """

    def __init__(self, activation="tanh", **kw):
        FFBase.__init__(self, 1, activation, **kw)
        self.gates = None

    def connect(self, brain):
        self.neurons = np.prod(inshape)
        self.weights = white(self.neurons, self.neurons*3)
        self.biases = zX(self.neurons*3)
        FFBase.connect()

    def feedforward(self, X) -> np.ndarray:
        self.inputs = rtm(X)
        self.gates = self.inputs.dot(self.weights) + self.biases
        self.gates[:, :self.neurons] = self.activation.forward(self.gates[:, :self.neurons])
        self.gates[:, self.neurons:] = sigmoid.forward(self.gates[:, self.neurons:])
        h, t, c = np.split(self.gates, 3, axis=1)
        self.output = h * t + self.inputs * c
        return self.output.reshape(X.shape)

    def backpropagate(self, delta) -> np.ndarray:
        shape = delta.shape
        delta = rtm(delta)

        h, t, c = np.split(self.gates, 3, axis=1)

        dh = self.activation.backward(h) * t * delta
        dt = sigmoid.backward(t) * h * delta
        dc = sigmoid.backward(c) * self.inputs * delta
        dx = c * delta

        dgates = np.concatenate((dh, dt, dc), axis=1)
        self.nabla_w = self.inputs.T.dot(dgates)
        self.nabla_b = dgates.sum(axis=0)

        return (dgates.dot(self.weights.T) + dx).reshape(shape)

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "Highway-{}".format(str(self.activation))


class DropOut(NoParamMixin, LayerBase):

    def __init__(self, dropchance):
        super().__init__()
        self.dropchance = scalX(1. - dropchance)
        self.mask = None
        self.inshape = None
        self.training = True

    def connect(self, brain):
        self.inshape = brain.outshape
        super().connect(brain)

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        self.mask = np.random.uniform(0, 1, self.inshape) < self.dropchance  # type: np.ndarray
        self.mask.astype(floatX)
        self.output = X * (self.mask if self.brain.learning else self.dropchance)
        return self.output

    def backpropagate(self, delta: np.ndarray) -> np.ndarray:
        output = delta * self.mask
        self.mask = np.ones_like(self.mask) * self.dropchance
        return output

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "DropOut({})".format(self.dropchance)
