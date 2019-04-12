from brainforge import backend as xp

from .abstract_layer import Layer, NoParamMixin, Parameterized
from .. import activations


class InputLayer(Layer):

    def __init__(self, shape, **kw):
        super().__init__(**kw)
        self.inshape = shape

    def connect(self, brain):
        self.brain = brain

    def forward(self, x: xp.ndarray):
        return x

    def backward(self, delta: xp.ndarray):
        return delta

    def outshape(self):
        return self.inshape


class Linear(Parameterized):

    def connect(self, brain):
        inshape = brain.outshape
        if len(inshape) != 1:
            err = "Dense only accepts input shapes with 1 dimension!\n"
            err += "Maybe you should consider placing <Flatten> before <Dense>?"
            raise RuntimeError(err)
        self.weights = self.initializer(inshape[0], self.units)
        self.biases = xp.zeros(self.units)
        super().connect(brain)

    def forward(self, x):
        self.inputs = x
        return x @ self.weights + self.biases

    def backward(self, delta):
        self.nabla_w = self.inputs.T @ delta
        self.nabla_b = xp.sum(delta, axis=1)
        return delta @ self.weights.T


class Activation(NoParamMixin, Layer):

    def __init__(self, activation="linear"):
        super().__init__()
        if isinstance(activation, activations.ActivationFunction):
            self.activation = activation
        elif isinstance(activation, str):
            self.activation = activations.get(activation)
        else:
            assert False

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        self.output = self.activation(x)
        return self.output

    def backward(self, delta) -> xp.ndarray:
        return delta * self.activation.derivative(self.output)

    @property
    def outshape(self):
        return self.inshape


class Reshape(NoParamMixin, Layer):

    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape

    def connect(self, brain):
        if self.shape is None:
            self.shape = xp.prod(brain.outshape),
        super().connect(brain)

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        return x.reshape(-1, *self.shape)

    def backward(self, delta: xp.ndarray) -> xp.ndarray:
        return delta.reshape(-1, self.inshape)

    @property
    def outshape(self):
        return self.shape


class Flatten(Reshape):
    pass
