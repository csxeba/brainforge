import numpy as np

from .abstract_model import Model


class LayerStack(Model):

    def __init__(self, input_shape, layers=()):
        super().__init__(input_shape)
        self.layers = []
        self.architecture = []
        self.learning = False
        self._iterme = None

        self._add_input_layer(input_shape)
        for layer in layers:
            self.add(layer)

    def _add_input_layer(self, input_shape):
        if isinstance(input_shape, int) or isinstance(input_shape, np.int64):
            input_shape = (input_shape,)
        from ..layers import InputLayer
        inl = InputLayer()
        inl.connect(to=self, inshape=input_shape)
        self.layers.append(inl)
        self.architecture.append(str(inl))

    def add(self, layer):
        layer.connect(self, inshape=self.layers[-1].outshape)
        self.layers.append(layer)
        self.architecture.append(str(layer))

    def pop(self):
        self.layers.pop()
        self.architecture.pop()

    def feedforward(self, X):
        for layer in self.layers:
            assert X.dtype == "float64"
            X = layer.feedforward(X)
        assert X.dtype == "float64"
        return X

    def get_weights(self, unfold=True):
        ws = [layer.get_weights(unfold=unfold) for
              layer in self.layers if layer.trainable]
        return np.concatenate(ws) if unfold else ws

    def set_weights(self, ws, fold=True):
        trl = (l for l in self.layers if l.trainable)
        if fold:
            start = 0
            for layer in trl:
                end = start + layer.nparams
                layer.set_weights(ws[start:end])
                start = end
        else:
            for w, layer in zip(ws, trl):
                layer.set_weights(w)

    def describe(self):
        return "Architecture: " + "->".join(self.architecture),

    def reset(self):
        for layer in (l for l in self.layers if l.trainable):
            layer.reset()

    @property
    def outshape(self):
        return self.layers[-1].outshape

    @property
    def nparams(self):
        return sum(layer.nparams for layer in self.layers if layer.trainable)

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterme is None:
            self._iterme = iter(self.layers)
        try:
            # noinspection PyTypeChecker
            return next(self._iterme)
        except StopIteration:
            self._iterme = None
            raise

    def __getitem__(self, item):
        return self.layers.__getitem__(item)
