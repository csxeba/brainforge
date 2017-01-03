import abc

import numpy as np
from util import white, white_like


class Layer(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, activation, **kw):

        from ..ops import act_fns

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
            self.activation = act_fns[activation]
        else:
            self.activation = activation

        if "trainable" in kw:
            self.trainable = kw["trainable"]
        else:
            self.trainable = True

    def connect(self, to, inshape):
        self.brain = to
        self.inshape = inshape

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


class _FFLayer(Layer):
    """Base class for the fully connected layer types"""
    def __init__(self, neurons: int, activation, **kw):
        Layer.__init__(self, activation, **kw)
        self.neurons = neurons

    @abc.abstractmethod
    def connect(self, to, inshape):
        Layer.connect(self, to, inshape)
        self.nabla_w = np.zeros_like(self.weights)
        self.nabla_b = np.zeros_like(self.biases)

    @property
    def outshape(self):
        return self.neurons if isinstance(self.neurons, tuple) else (self.neurons,)


class _Op(Layer):

    def __init__(self):
        Layer.__init__(self, activation="linear", trainable=False)
        self.opf = None
        self.opb = None

    def connect(self, to, inshape):
        Layer.connect(self, to, inshape)

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        self.output = self.opf(stimuli)
        return self.output

    def backpropagate(self, error) -> np.ndarray:
        return self.opb(error)

    def get_weights(self, unfold=True):
        return NotImplemented

    def set_weights(self, w, fold=True):
        return NotImplemented

    def shuffle(self) -> None:
        return NotImplemented

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


class Activation(Layer):

    def __init__(self, activation):
        Layer.__init__(self, activation, trainable=False)

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        self.output = self.activation(stimuli)
        return self.output

    def backpropagate(self, error) -> np.ndarray:
        return error * self.activation.derivative(self.output)

    @property
    def outshape(self):
        return self.inshape

    def capsule(self):
        return Layer.capsule(self) + [self.activation]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(activation=capsule[-1])

    def __str__(self):
        return "Activation-{}".format(str(self.activation))


class InputLayer(Layer):

    def __init__(self, shape):
        Layer.__init__(self, activation="linear", trainable=False)
        self.neurons = shape

    def connect(self, to, inshape):
        Layer.connect(self, to, inshape)
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

    def shuffle(self): pass

    def get_weights(self, unfold=True):
        return None

    def set_weights(self, w, fold=True):
        pass

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


class DenseLayer(_FFLayer):
    """Just your regular Densely Connected Layer

    Aka Dense (Keras), Fully Connected, Feedforward, etc.
    Elementary building block of the Multilayer Perceptron.
    """

    def __init__(self, neurons, activation="linear", **kw):
        if isinstance(neurons, tuple):
            neurons = neurons[0]
        _FFLayer.__init__(self,  neurons=neurons, activation=activation, **kw)

    def connect(self, to, inshape):
        if len(inshape) != 1:
            err = "Dense only accepts input shapes with 1 dimension!\n"
            err += "Maybe you should consider placing <Flatten> before <Dense>?"
            raise RuntimeError(err)
        self.weights = white(inshape[0], self.neurons)
        self.biases = np.zeros((self.neurons,))
        _FFLayer.connect(self, to, inshape)

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
        return _FFLayer.capsule(self) + [self.activation, self.get_weights(unfold=False)]

    @classmethod
    def from_capsule(cls, capsule):
        return cls(neurons=capsule[-1][0].shape[1], activation=capsule[-2], trainable=capsule[1])

    def __str__(self):
        return "Dense-{}-{}".format(self.neurons, str(self.activation)[:5])


class Flatten(_Op):

    def connect(self, to, inshape):
        from ..ops import Flatten as Flat, Reshape as Resh
        _Op.connect(self, to, inshape)
        self.opf = Flat()
        self.opb = Resh(inshape)


class Reshape(_Op):

    def connect(self, to, inshape):
        from ..ops import Flatten as Flat, Reshape as Resh
        _Op.connect(self, to, inshape)
        self.opf = Resh(inshape)
        self.opb = Flat()


