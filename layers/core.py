import numpy as np

from .. import atomic
from ..util import white, zX
from .abstract_layer import LayerBase, NoParamMixin, FFBase


class DenseLayer(FFBase):

    def connect(self, to, inshape):
        if len(inshape) != 1:
            err = "Dense only accepts input shapes with 1 dimension!\n"
            err += "Maybe you should consider placing <Flatten> before <Dense>?"
            raise RuntimeError(err)
        self.weights = white(inshape[0], self.neurons)
        self.biases = zX(self.neurons)
        super().connect(to, inshape)

    def feedforward(self, X):
        self.inputs = X
        self.output = self.activation(atomic.DenseOp.forward(
            X, self.weights, self.biases
        ))
        return self.output

    def backpropagate(self, delta):
        delta *= self.activation.derivative(self.output)
        self.nabla_w, self.nabla_b, dX = atomic.DenseOp.backward(
            self.inputs, delta, self.weights
        )
        return dX

    def capsule(self):
        return FFBase.capsule(self) + [self.activation, self.get_weights(unfold=False)]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(neurons=capsule[-1][0].shape[1], activation=capsule[-2])

    def __str__(self):
        return "Dense-{}-{}".format(self.neurons, str(self.activation)[:5])


class Activation(NoParamMixin, LayerBase):

    def __init__(self, activation="linear"):
        LayerBase.__init__(self, activation)

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        self.output = self.activation(stimuli)
        return self.output

    def backpropagate(self, error) -> np.ndarray:
        if self.position > 1:
            return error * self.activation.derivative(self.output)

    @property
    def outshape(self):
        return self.inshape

    def capsule(self):
        return LayerBase.capsule(self) + [self.activation]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(activation=capsule[-1])

    def __str__(self):
        return "Activation-{}".format(str(self.activation))


class InputLayer(Activation):

    def __init__(self):
        super().__init__("linear")


class Reshape(LayerBase, NoParamMixin):

    def __init__(self, shape=None):
        LayerBase.__init__(self, activation="linear", trainable=False)
        self.shape = shape

    def connect(self, to, inshape):
        if self.shape is None:
            self.shape = np.prod(inshape),
        super().connect(to, inshape)

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        return atomic.ReshapeOp.forward(stimuli, self.shape)

    def backpropagate(self, error) -> np.ndarray:
        return atomic.ReshapeOp.forward(error, self.inshape)

    def capsule(self):
        return [self.inshape]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(*capsule)

    def outshape(self):
        return self.shape

    def __str__(self):
        return self.__class__.__name__


class Flatten(Reshape):

    def __init__(self):
        super().__init__(None)
