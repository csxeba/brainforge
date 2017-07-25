import numpy as np

from .abstract_layer import LayerBase, NoParamMixin, FFBase
from ..atomic import Sigmoid
from ..util import rtm, scalX, zX, white
from ..config import floatX

sigmoid = Sigmoid()


class HighwayLayer(FFBase):
    """
    Neural Highway Layer based on Srivastava et al., 2015
    """

    def __init__(self, activation="tanh", **kw):
        FFBase.__init__(self, 1, activation, **kw)
        self.gates = None

    def connect(self, to, inshape):
        self.neurons = np.prod(inshape)
        self.weights = white(self.neurons, self.neurons*3)
        self.biases = zX(self.neurons*3)
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

    def connect(self, to, inshape):
        self.inshape = inshape
        super().connect(to, inshape)

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        self.inputs = stimuli
        self.mask = np.random.uniform(0, 1, self.inshape) < self.dropchance  # type: np.ndarray
        self.mask.astype(floatX)
        self.output = stimuli * (self.mask if self.brain.learning else self.dropchance)
        return self.output

    def backpropagate(self, error: np.ndarray) -> np.ndarray:
        output = error * self.mask
        self.mask = np.ones_like(self.mask) * self.dropchance
        return output

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "DropOut({})".format(self.dropchance)
