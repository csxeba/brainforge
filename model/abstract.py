import abc

import numpy as np


class GraphicalModel(abc.ABC):

    def __init__(self, input_shape, layers=(), name=""):
        # Referencing the data wrapper on which we do the learning
        self.name = name
        # Containers and self-describing variables
        self.layers = []
        self.architecture = []
        self.age = 0
        self.cost = None
        self.optimizer = None

        self.X = None
        self.Y = None
        self.N = 0  # X's size goes here
        self.m = 0  # Batch size goes here
        self._finalized = False
        self.learning = False

        self._add_input_layer(input_shape)
        if layers:
            from ..architecture import LayerBase
            for layer in layers:
                if not isinstance(layer, LayerBase):
                    raise RuntimeError("Supplied layer is not an instance of LayerBase!\n"+str(layer))
                self.add(layer)

    def _add_input_layer(self, input_shape):
        if not input_shape:
            raise RuntimeError("Parameter input_shape must be supplied for the first layer!")
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        from ..architecture import InputLayer
        inl = InputLayer(input_shape)
        inl.connect(to=self, inshape=input_shape)
        inl.connected = True
        self.layers.append(inl)
        self.architecture.append(str(inl))

    def add(self, layer):
        layer.connect(self, inshape=self.layers[-1].outshape)
        self.layers.append(layer)
        self.architecture.append(str(layer))
        layer.connected = True
        self._finalized = False

    def pop(self):
        self.layers.pop()
        self.architecture.pop()
        self._finalized = False

    def fit_generator(self, generator, lessons_per_epoch, epochs=30, monitor=(), validation=(), verbose=1):
        self.N = lessons_per_epoch

        epcosts = []
        lstr = len(str(epochs))
        epoch = 1
        while epoch <= epochs:
            if verbose:
                print("Epoch {:>{w}}/{}".format(epoch, epochs, w=lstr))

            epcosts += self.epoch(generator, monitor, validation, verbose)
            epoch += 1

        self.age += epochs
        return epcosts

    def fit(self, X, Y, batch_size=20, epochs=30, monitor=(), validation=(), verbose=1, shuffle=True):
        datastream = self._batch_stream(X, Y, batch_size, shuffle)
        return self.fit_generator(datastream, len(X), epochs, monitor, validation, verbose)

    def epoch(self, generator, monitor, validation, verbose):

        if not self._finalized:
            raise RuntimeError("Architecture not finalized!")

        costs = []
        done = 0.

        self.learning = True
        while round(done, 5) < 1.:
            cost = self.learn_batch(*next(generator))
            cost /= self.m
            costs.append(cost)

            done += self.m / self.N
            if verbose:
                print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(done, np.mean(costs)), end="")
        self.learning = False

        if verbose:
            print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(1., np.mean(costs)), end="")

        if verbose:
            if validation:
                self._print_progress(validation, monitor)
            print()

        return costs

    def classify(self, X):
        return np.argmax(self.predict(X), axis=1)

    def predict(self, X):
        self.m = X.shape[0]
        for layer in self.layers:
            X = layer.feedforward(X)
        return X

    def evaluate(self, X, Y, batch_size=32, classify=True, shuffle=False, verbose=False):
        if not batch_size or batch_size == "full":
            batch_size = len(X)
        N = X.shape[0]
        batches = self._batch_stream(X, Y, batch_size, shuffle, infinite=False)

        cost = []
        acc = []
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
        if verbose:
            print("\rEvaluating: {:>7.2%}".format(1.))
        if classify:
            return np.mean(cost), np.mean(acc)
        return np.mean(cost)

    @staticmethod
    def _batch_stream(X, Y, m, shuffle=True, infinite=True):
        arg = np.arange(X.shape[0])
        while 1:
            if shuffle:
                np.random.shuffle(arg)
                X, Y = X[arg], Y[arg]
            for x, y in ((X[start:start + m], Y[start:start + m])
                         for start in range(0, X.shape[0], m)):
                yield x, y
            if not infinite:
                break

    def get_weights(self, unfold=True):
        ws = [
            layer.get_weights(unfold=unfold) for
            layer in self.layers if layer.trainable
        ]
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
    def weights(self):
        return self.get_weights(unfold=False)

    @weights.setter
    def weights(self, ws):
        self.set_weights(ws, fold=(ws.ndim > 1))

    @abc.abstractmethod
    def learn_batch(self, X, Y):
        raise NotImplementedError

    @property
    def nparams(self):
        return sum(layer.nparams for layer in self.layers if layer.trainable)

