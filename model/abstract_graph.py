import abc

import numpy as np

from ..util import batch_stream


class Graph(abc.ABC):

    def __init__(self, input_shape, layers=(), name=""):
        self.name = name
        self.layers = []
        self.architecture = []
        self.cost = None

        self.N = 0  # X's size goes here
        self.m = 0  # Batch size goes here
        self._finalized = False

        self._add_input_layer(input_shape)
        for layer in layers:
            self.add(layer)

    def _add_input_layer(self, input_shape):
        if isinstance(input_shape, int) or isinstance(input_shape, np.int64):
            input_shape = (input_shape,)
        from ..layers import InputLayer
        inl = InputLayer(input_shape)
        inl.connect(to=self, inshape=input_shape)
        self.layers.append(inl)
        self.architecture.append(str(inl))

    def add(self, layer):
        layer.connect(self, inshape=self.layers[-1].outshape)
        self.layers.append(layer)
        self.architecture.append(str(layer))
        self._finalized = False

    def pop(self):
        self.layers.pop()
        self.architecture.pop()
        self._finalized = False

    def finalize(self, cost, **kw):
        from ..cost import cost_functions
        self.cost = cost_functions[cost] \
            if isinstance(cost, str) else cost
        return self

    def classify(self, X):
        return np.argmax(self.predict(X), axis=1)

    def predict(self, X):
        self.m = X.shape[0]
        for layer in self.layers:
            X = layer.feedforward(X)
        return X

    def evaluate(self, X, Y, batch_size=32, classify=True, shuffle=False, verbose=False):
        N = X.shape[0]
        batches = batch_stream(X, Y, m=batch_size, shuffle=shuffle, infinite=False)

        cost, acc = [], []
        for m, (x, y) in enumerate(batches, start=1):
            if verbose:
                print("\rEvaluating: {:>7.2%}".format((m*batch_size) / N), end="")
            pred = self.predict(x)
            cost.append(self.cost(pred, y) / self.m)
            if classify:
                pred_classes = np.argmax(pred, axis=1)
                trgt_classes = np.argmax(y, axis=1)
                eq = np.equal(pred_classes, trgt_classes)
                acc.append(eq.mean())
        results = np.mean(cost)
        if classify:
            results = (results, np.mean(acc))
        return results

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

    @property
    def nparams(self):
        return sum(layer.nparams for layer in self.layers if layer.trainable)
