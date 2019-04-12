from brainforge import backend as xp
from ..parameter import initializers

from ..util import white_like, zX_like


class Layer:

    trainable = True

    def __init__(self, **kw):

        self.position = 0
        self.brain = None
        self.inputs = None
        self.output = None
        self.inshape = None
        self.connected = False

    def connect(self, brain):
        self.brain = brain
        self.inshape = brain.outshape
        self.position = len(self.brain.layers)
        self.connected = True

    def forward(self, x: xp.ndarray) -> xp.ndarray: raise NotImplementedError

    def backward(self, d: xp.ndarray) -> xp.ndarray: raise NotImplementedError

    @property
    def outshape(self): raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class NoParamMixin:

    trainable = False

    def shuffle(self): pass

    def get_weights(self, unfold=True): pass

    def set_weights(self, w, fold=True): pass

    def gradients(self): pass

    @property
    def nparams(self):
        return 0


class Parameterized(Layer):

    """Base class for the fully connected layer types"""

    def __init__(self, units, weight_initializer="glorot_normal", **kw):
        super().__init__(**kw)
        self.units = int(units)
        self.weights = None
        self.biases = None
        self.nabla_w = None
        self.nabla_b = None
        self.initializer = initializers[weight_initializer]

    def connect(self, brain):
        super().connect(brain)
        self.nabla_w = xp.empty_like(self.weights)
        self.nabla_b = xp.empty_like(self.biases)

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        raise NotImplementedError

    def backward(self, x: xp.ndarray) -> xp.ndarray:
        raise NotImplementedError

    def get_weights(self, unfold=True):
        if unfold:
            return xp.concatenate((self.weights.ravel(), self.biases.ravel()))
        return [self.weights, self.biases]

    def set_weights(self, w, fold=True):
        if fold:
            W = self.weights
            self.weights = w[:W.size].reshape(W.shape)
            self.biases = w[W.size:].reshape(self.biases.shape)
        else:
            self.weights, self.biases = w

    @property
    def outshape(self) -> tuple:
        return self.units,
