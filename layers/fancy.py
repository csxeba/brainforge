import numpy as np
from layers import Layer
from layers.core import _FFLayer
from ops import sigmoid
from util import white, rtm


class HighwayLayer(_FFLayer):
    """
    Neural Highway Layer

    Based on Srivastava et al., 2015

    A carry gate is applied to the raw input.
    A transform gate is applied to the output activation.
    y = y_ * g_t + x * g_c
    Output shape equals the input shape.
    """

    def __init__(self, activation="tanh", **kw):
        _FFLayer.__init__(self, 1, activation, **kw)
        self.gates = None

    def connect(self, to, inshape):
        self.neurons = int(np.prod(inshape))
        self.weights = white(self.neurons, self.neurons*3)
        self.biases = np.zeros((self.neurons*3,))
        _FFLayer.connect(self, to, inshape)

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
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
        return _FFLayer.capsule(self) + [self.activation, self.get_weights(unfold=False)]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(activation=capsule[-2])

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "Highway-{}".format(str(self.activation))


class DropOut(Layer):

    def __init__(self, dropchance):
        Layer.__init__(self, activation="linear", trainable=False)
        self.dropchance = 1. - dropchance
        self.mask = None
        self.neurons = None

    def connect(self, to, inshape):
        self.neurons = inshape

    def feedforward(self, questions):
        self.inputs = questions
        self.mask = np.random.uniform(0, 1, self.neurons) < self.dropchance
        self.output = questions * self.mask
        return self.output

    def backpropagate(self, error):
        output = error * self.mask
        self.mask = np.ones_like(self.mask) * self.dropchance
        return output

    def get_weights(self, unfold=True): raise NotImplementedError

    def set_weights(self, w, fold=True): raise NotImplementedError

    def shuffle(self) -> None: raise NotImplementedError

    @property
    def outshape(self):
        return self.neurons

    def capsule(self):
        return Layer.capsule(self) + [self.dropchance]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(dropchance=capsule[-1])

    def __str__(self):
        return "DropOut({})".format(self.dropchance)


class Experimental:

    class AboLayer(Layer):
        def __init__(self, brain, position, activation):
            Layer.__init__(self, brain, position, activation)
            self.brain = brain
            self.fanin = brain.layers[-1].fanout
            self.neurons = []

        def add_minion(self, empty_network):
            minion = empty_network
            minion.add_fc(10)
            minion.finalize_architecture()
            self.neurons.append(minion)

        def feedforward(self, inputs):
            """this ain't so simple after all O.O"""
            pass

        def receive_error(self, error_vector: np.ndarray) -> None:
            pass

        def shuffle(self) -> None:
            pass

        def backpropagate(self, error) -> np.ndarray:
            pass

        def weight_update(self) -> None:
            pass

        def predict(self, stimuli: np.ndarray) -> np.ndarray:
            pass

        def outshape(self):
            return ...