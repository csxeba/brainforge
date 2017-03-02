import abc

import numpy as np

from ..util import white, white_like


class LayerBase(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, activation, **kw):

        from brainforge.ops import act_fns

        self.position = 0
        self.brain = None
        self.inputs = None
        self.output = None
        self.inshape = None

        self.weights = None
        self.biases = None
        self.nabla_w = None
        self.nabla_b = None

        self.connected = False

        self.optimizer = None

        if isinstance(activation, str):
            self.activation = act_fns[activation]()
        else:
            self.activation = activation

        if "trainable" in kw:
            self.trainable = kw["trainable"]
        else:
            self.trainable = True

    def connect(self, to, inshape):
        self.brain = to
        self.inshape = inshape
        self.position = len(self.brain.layers)

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray) -> np.ndarray: raise NotImplementedError

    @abc.abstractmethod
    def backpropagate(self, error) -> np.ndarray: raise NotImplementedError

    def shuffle(self) -> None:
        self.weights = white_like(self.weights)
        self.biases = np.zeros_like(self.biases)

    def get_weights(self, unfold=True):
        if unfold:
            return np.concatenate((self.weights.ravel(), self.biases.ravel()))
        return [self.weights, self.biases]

    def set_weights(self, w, fold=True):
        if fold:
            sw = self.weights
            self.weights = w[:sw.size].reshape(sw.shape)
            self.biases = w[sw.size:].reshape(self.biases.shape)
        else:
            self.weights, self.biases = w

    @property
    def gradients(self):
        return np.concatenate([self.nabla_w.ravel(), self.nabla_b.ravel()])

    @property
    def nparams(self):
        return self.weights.size + self.biases.size

    def capsule(self):
        return [self.inshape, self.trainable]

    @classmethod
    @abc.abstractmethod
    def from_capsule(cls, capsule):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def outshape(self): raise NotImplementedError

    @abc.abstractmethod
    def __str__(self): raise NotImplementedError


class NoParamMixin(abc.ABC):

    def shuffle(self) -> None:
        pass

    def get_weights(self, unfold=True):
        pass

    def set_weights(self, w, fold=True):
        pass

    def gradients(self):
        pass

    @property
    def nparams(self):
        return


class FFBase(LayerBase):
    """Base class for the fully connected layer types"""
    def __init__(self, neurons: int, activation, **kw):
        LayerBase.__init__(self, activation, **kw)
        self.neurons = neurons

    @abc.abstractmethod
    def connect(self, to, inshape):
        LayerBase.connect(self, to, inshape)
        self.nabla_w = np.zeros_like(self.weights)
        self.nabla_b = np.zeros_like(self.biases)

    @property
    def outshape(self):
        return self.neurons if isinstance(self.neurons, tuple) else (self.neurons,)


class _Op(LayerBase, NoParamMixin):

    def __init__(self):
        LayerBase.__init__(self, activation="linear", trainable=False)
        self.opf = None
        self.opb = None

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        self.output = self.opf(stimuli)
        return self.output

    def backpropagate(self, error) -> np.ndarray:
        return self.opb(error)

    def capsule(self):
        return [self.inshape]

    @classmethod
    def from_capsule(cls, capsule):
        return cls()

    @property
    def outshape(self):
        return self.opf.outshape(self.inshape)

    def __str__(self):
        return str(self.opf)


class Activation(LayerBase, NoParamMixin):

    def __init__(self, activation):
        LayerBase.__init__(self, activation, trainable=False)

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


class InputLayer(LayerBase, NoParamMixin):

    def __init__(self, shape):
        LayerBase.__init__(self, activation="linear", trainable=False)
        self.neurons = shape

    def connect(self, to, inshape):
        LayerBase.connect(self, to, inshape)
        assert inshape == self.neurons

    def feedforward(self, questions):
        """
        Passes the unmodified input matrix

        :param questions: numpy.ndarray
        :return: numpy.ndarray
        """
        self.inputs = self.output = questions
        return questions

    def backpropagate(self, error): pass

    def capsule(self):
        return [self.inshape]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(shape=capsule[0])

    @property
    def outshape(self):
        return self.neurons if isinstance(self.neurons, tuple) else (self.neurons,)

    def __str__(self):
        return "Input-{}".format(self.neurons)


class DenseLayer(FFBase):
    """Just your regular Densely Connected Layer

    Aka Dense (Keras), Fully Connected, Feedforward, etc.
    Elementary building block of the Multilayer Perceptron.
    """

    def __init__(self, neurons, activation="linear", **kw):
        if isinstance(neurons, tuple):
            neurons = neurons[0]
        FFBase.__init__(self, neurons=neurons, activation=activation, **kw)

    def connect(self, to, inshape):
        if len(inshape) != 1:
            err = "Dense only accepts input shapes with 1 dimension!\n"
            err += "Maybe you should consider placing <Flatten> before <Dense>?"
            raise RuntimeError(err)
        self.weights = white(inshape[0], self.neurons)
        self.biases = np.zeros((self.neurons,))
        FFBase.connect(self, to, inshape)

    def feedforward(self, questions):
        """
        Transforms the input matrix with a weight matrix.

        :param questions: numpy.ndarray of shape (lessons, prev_layer_output)
        :return: numpy.ndarray: transformed matrix
        """
        self.inputs = questions
        self.output = self.activation(np.dot(questions, self.weights) + self.biases)
        return self.output

    def backpropagate(self, error):
        """
        Backpropagates the errors.
        Calculates gradients of the weights, then
        returns the previous layer's error.

        :param error:
        :return: numpy.ndarray
        """
        error *= self.activation.derivative(self.output)
        self.nabla_w = np.dot(self.inputs.T, error)
        self.nabla_b = np.sum(error, axis=0)
        return np.dot(error, self.weights.T)

    def capsule(self):
        return FFBase.capsule(self) + [self.activation, self.get_weights(unfold=False)]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(neurons=capsule[-1][0].shape[1], activation=capsule[-2], trainable=capsule[1])

    def __str__(self):
        return "Dense-{}-{}".format(self.neurons, str(self.activation)[:5])


class Reshape(_Op):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def connect(self, to, inshape):
        from brainforge.ops import ReshapeOp
        _Op.connect(self, to, inshape)
        self.opf = ReshapeOp(self.shape)
        self.opb = ReshapeOp(inshape)

    def backpropagate(self, error):
        if self.position > 1:
            return super().backpropagate(error)


class Flatten(Reshape):

    def __init__(self):
        super().__init__(None)

    def connect(self, to, inshape):
        from brainforge.ops import ReshapeOp
        super().connect(to, inshape)
        self.opf = ReshapeOp((np.prod(inshape),))
