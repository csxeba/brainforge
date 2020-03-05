import abc


class Model(abc.ABC):

    def __init__(self, input_shape, **kw):
        self.input_shape = input_shape
        self.floatX = kw.get("floatX", "float64")
        self.compiled = kw.get("compiled", False)

    @abc.abstractmethod
    def feedforward(self, X):
        raise NotImplementedError

    @abc.abstractmethod
    def get_weights(self, unfold):
        raise NotImplementedError

    @abc.abstractmethod
    def set_weights(self, fold):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_params(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def outshape(self):
        raise NotImplementedError
