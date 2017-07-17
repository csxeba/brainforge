import numpy as np

from .core import LayerBase, FFBase, NoParamMixin
from ..ops import Sigmoid
from ..util import white, rtm, zX, floatX, scalX

sigmoid = Sigmoid()


class HighwayLayer(FFBase):
    """
    Neural Highway Layer

    Based on Srivastava et al., 2015

    A carry gate is applied to the raw input.
    A transform gate is applied to the output activation.
    y = y_ * g_t + x * g_c
    Output shape equals the input shape.
    """

    def __init__(self, activation="tanh", **kw):
        FFBase.__init__(self, 1, activation, **kw)
        self.gates = None

    def connect(self, to, inshape):
        self.neurons = np.prod(inshape)
        self.weights = white(self.neurons, self.neurons*3)
        self.biases = zX(self.neurons*3,)
        FFBase.connect(self, to, inshape)

    def feedforward(self, stimuli) -> np.ndarray:
        self.inputs = rtm(stimuli)
        self.gates = self.inputs.dot(self.weights) + self.biases
        self.gates[:, :self.neurons] = self.activation(self.gates[:, :self.neurons])
        self.gates[:, self.neurons:] = sigmoid(self.gates[:, self.neurons:])
        h, t, c = np.split(self.gates, 3, axis=1)
        self.output = h * t + self.inputs * c
        return self.output.reshape(stimuli.shape)

    def backpropagate(self, error) -> np.ndarray:
        shape = error.shape
        error = rtm(error)

        h, t, c = np.split(self.gates, 3, axis=1)

        dh = self.activation.derivative(h) * t * error
        dt = sigmoid.derivative(t) * h * error
        dc = sigmoid.derivative(c) * self.inputs * error
        dx = c * error

        dgates = np.concatenate((dh, dt, dc), axis=1)
        self.nabla_w = self.inputs.T.dot(dgates)
        self.nabla_b = dgates.sum(axis=0)

        return (dgates.dot(self.weights.T) + dx).reshape(shape)

    def capsule(self):
        return FFBase.capsule(self) + [self.activation, self.get_weights(unfold=False)]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(activation=capsule[-2])

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "Highway-{}".format(str(self.activation))


class DropOut(LayerBase, NoParamMixin):

    def __init__(self, dropchance):
        LayerBase.__init__(self, activation="linear", trainable=False)
        self.dropchance = scalX(1. - dropchance)
        self.mask = None
        self.neurons = None
        self.training = True

    def connect(self, to, inshape):
        self.neurons = inshape
        LayerBase.connect(self, to, inshape)

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        self.inputs = stimuli
        self.mask = np.random.uniform(0, 1, self.neurons) < self.dropchance  # type: np.ndarray
        self.mask.astype(floatX)
        self.output = stimuli * (self.mask if self.brain.learning else self.dropchance)
        return self.output

    def backpropagate(self, error: np.ndarray) -> np.ndarray:
        output = error * self.mask
        self.mask = np.ones_like(self.mask, dtype=floatX) * self.dropchance
        return output

    @property
    def outshape(self):
        return self.neurons

    def capsule(self):
        return LayerBase.capsule(self) + [self.dropchance]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(dropchance=capsule[-1])

    def __str__(self):
        return "DropOut({})".format(self.dropchance)
