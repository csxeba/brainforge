"""
Neural Network Framework on top of NumPy
Copyright (C) 2016  Csaba Gór

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import warnings

import numpy as np


class Network:

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
            from ..layers import LayerBase
            for layer in layers:
                if not isinstance(layer, LayerBase):
                    raise RuntimeError("Supplied layer is not an instance of LayerBase!\n"+str(layer))
                self.add(layer)

    def encapsulate(self, dumppath=None):
        from ..util.persistance import Capsule
        capsule = Capsule(**{
            "flpath": dumppath,
            "name": self.name,
            "cost": self.cost,
            "optimizers": self.optimizer,
            "architectures": self.architecture[:],
            "layers": [layer.capsule() for layer in self.layers]})

        if dumppath is not None:
            capsule.dump()
        return capsule

    @classmethod
    def from_capsule(cls, capsule):

        from ..optimizers import optimizers
        from ..util.persistance import Capsule
        from ..util.shame import translate_architecture as trsl

        if not isinstance(capsule, Capsule):
            capsule = Capsule.read(capsule)
        c = capsule

        net = Network(input_shape=c["layers"][0][0], name=c["name"])

        for layer_name, layer_capsule in zip(c["architectures"], c["layers"]):
            if layer_name[:5] == "Input":
                continue
            layer_cls = trsl(layer_name)
            layer = layer_cls.from_capsule(layer_capsule)
            net.add(layer)

        opti = c["optimizers"]
        if isinstance(opti, str):
            opti = optimizers[opti]()
        net.finalize(cost=c["cost"], optimizer=opti)

        for layer, lcaps in zip(net.layers, c["layers"]):
            if layer.weights is not None:
                layer.set_weights(lcaps[-1], fold=False)

        return net

    @classmethod
    def from_csxdata(cls, frame, layers=(), name=""):
        inshp = frame.neurons_required[0]
        return cls(inshp, layers, name)

    # ---- Methods for architecture building ----

    def _add_input_layer(self, input_shape):
        if not input_shape:
            raise RuntimeError("Parameter input_shape must be supplied for the first layer!")
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        from ..layers import InputLayer
        inl = InputLayer(input_shape)
        inl.connect(to=self, inshape=input_shape)
        inl.connected = True
        self.layers.append(inl)
        self.architecture.append(str(inl))

    def add(self, layer, input_dim=()):
        if len(self.layers) == 0:
            self._add_input_layer(input_dim)
            self.architecture.append(str(self.layers[-1]))

        layer.connect(self, inshape=self.layers[-1].outshape)
        self.layers.append(layer)
        self.architecture.append(str(layer))
        layer.connected = True

    def finalize(self, cost="mse", optimizer="sgd"):
        from ..costs import cost_functions
        from ..optimizers import optimizers

        self.cost = cost_functions[cost] \
            if isinstance(cost, str) else cost
        self.optimizer = optimizers[optimizer](self.nparams) \
            if isinstance(optimizer, str) else optimizer
        self._finalized = True

    def pop(self):
        self.layers.pop()
        self.architecture.pop()
        self._finalized = False

    # ---- Methods for model fitting ----

    def fit(self, X, Y, batch_size=20, epochs=30, monitor=(), validation=(), verbose=1, shuffle=True):
        self.N = X.shape[0]

        costs = []
        lstr = len(str(epochs))
        for epoch in range(1, epochs+1):
            if verbose:
                print("Epoch {:>{w}}/{}".format(epoch, epochs, w=lstr))
            batches = self._batch_stream(X, Y, batch_size, shuffle)
            costs += self.epoch(batches, monitor, validation, verbose)
        self.age += epochs
        return costs

    def fit_generator(self, generator, lessons_per_epoch, epochs=30, monitor=(), validation=(), verbose=1):
        self.N = epochs * lessons_per_epoch

        epcosts = []
        lstr = len(str(epochs))
        epoch = 1
        while epoch <= epochs:
            if verbose:
                print("\nEpoch {:>{w}}/{}".format(epoch, epochs, w=lstr))

            epcosts += self.epoch(generator, monitor, validation, verbose)
            epoch += 1

        self.age += epochs
        return epcosts

    def fit_csxdata(self, frame, batch_size=20, epochs=10, monitor=(), verbose=1, shuffle=True):
        fanin, outshape = frame.neurons_required
        if fanin != self.layers[0].outshape or outshape != self.layers[-1].outshape:
            errstring = "Network configuration incompatible with supplied dataframe!\n"
            errstring += "fanin: {} <-> InputLayer: {}\n".format(fanin, self.layers[0].outshape)
            errstring += "outshape: {} <-> Net outshape: {}\n".format(outshape, self.layers[-1].outshape)
            raise RuntimeError(errstring)

        validation = frame.table("testing") if frame.n_testing else ()
        batch_stream = frame.batchgen(batch_size, "learning", weigh=False, infinite=True)

        return self.fit_generator(batch_stream, frame.N // batch_size,
                                  epochs, monitor, validation, verbose)

    def epoch(self, generator, monitor, validation, verbose):

        if not self._finalized:
            raise RuntimeError("Architecture not finalized!")

        costs = []
        done = 0.

        self.learning = True
        while round(done, 5) < 1.:
            cost = self.learn_batch(*next(generator))
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

    def learn_batch(self, X, Y, parameter_update=True):
        self.X, self.Y = X, Y
        preds = self.prediction(self.X)
        delta = self.cost.derivative(preds, Y)
        self.backpropagation(delta)
        if parameter_update:
            self._parameter_update()

        return self.cost(self.output, self.Y)

    def _parameter_update(self):
        W = self.optimizer.optimize(
            self.get_weights(), self.get_gradients(), self.m
        )
        self.set_weights(W)

    def _print_progress(self, validation, monitor):
        classificaton = "acc" in monitor
        results = self.evaluate(*validation, classify=classificaton)

        chain = "testing cost: {0:.5f}"
        if classificaton:
            tcost, tacc = results
            accchain = "\taccuracy: {0:.2%}".format(tacc)
        else:
            tcost = results
            accchain = ""
        print(chain.format(tcost) + accchain, end="")

    # ---- Methods for forward/backward propagation ----

    def classify(self, X):
        return np.argmax(self.prediction(X), axis=1)

    def prediction(self, X):
        self.m = X.shape[0]
        for layer in self.layers:
            X = layer.feedforward(X)
            assert X.dtype == "float64", "Forward TypeCheck failed after " + str(layer)
        return X

    def predict_proba(self, X):
        return self.prediction(X)

    def evaluate(self, X, Y, batch_size=32, classify=True, shuffle=False):
        if not batch_size or batch_size == "full":
            batch_size = len(X)
        batches = self._batch_stream(X, Y, batch_size, shuffle)

        cost = []
        acc = []
        for x, y in batches:
            pred = self.prediction(x)
            cost.append(self.cost(pred, y) / y.shape[0])
            if classify:
                pred_classes = np.argmax(pred, axis=1)
                trgt_classes = np.argmax(y, axis=1)
                eq = np.equal(pred_classes, trgt_classes)
                acc.append(eq.mean())
        if classify:
            return np.mean(cost), np.mean(acc)
        return np.mean(cost)

    def backpropagation(self, error):
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)
        return error

    # ---- Some utilities ----

    @staticmethod
    def _batch_stream(X, Y, m, shuffle=True):
        if shuffle:
            arg = np.arange(X.shape[0])
            np.random.shuffle(arg)
            X, Y = X[arg], Y[arg]
        return (((X[start:start + m], Y[start:start + m])
                 for start in range(0, X.shape[0], m)))

    def shuffle(self):
        for layer in self.layers:
            if layer.trainable:
                layer.shuffle()

    def describe(self, verbose=0):
        if not self.name:
            name = "BrainForge Artificial Neural Network."
        else:
            name = "{}, the Artificial Neural Network.".format(self.name)
        chain = "----------\n"
        chain += name + "\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture: " + "->".join(self.architecture) + "\n"
        chain += "----------"
        if verbose:
            print(chain)
        else:
            return chain

    def get_weights(self, unfold=True):
        ws = [layer.get_weights(unfold=unfold) for layer in self.layers if layer.trainable]
        return np.concatenate(ws) if unfold else ws

    def set_weights(self, ws, fold=True):
        if fold:
            start = 0
            for layer in self.layers:
                if not layer.trainable:
                    continue
                end = start + layer.nparams
                layer.set_weights(ws[start:end])
                start = end
        else:
            for w, layer in zip(ws, self.layers):
                if not layer.trainable:
                    continue
                layer.set_weights(w)

    def get_gradients(self, unfold=True):
        grads = [l.gradients for l in self.layers if l.trainable]
        if unfold:
            grads = np.concatenate(grads)
        return grads

    def gradient_check(self, X, Y, verbose=1, epsilon=1e-5):
        from ..util import gradient_check
        if self.age == 0:
            warnings.warn("Performing gradient check on an untrained Neural Network!",
                          RuntimeWarning)
        return gradient_check(self, X, Y, verbose=verbose, epsilon=epsilon)

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def weights(self):
        return self.get_weights(unfold=False)

    @weights.setter
    def weights(self, ws):
        self.set_weights(ws, fold=(ws.ndim > 1))

    @property
    def gradients(self):
        return self.get_gradients(unfold=True)

    @property
    def nparams(self):
        return sum(layer.nparams for layer in self.layers if layer.trainable)


class Autoencoder(Network):

    def __init__(self, inshape=(), decoder_type="learnable", name=""):
        Network.__init__(self, inshape, name)
        if decoder_type not in ("learnable", "fixed", "built", "single", None):
            raise ValueError('decoder_type should be one of:\n"learnable", "mirrored", "built", "single", None')
        self.decoder_type = decoder_type
        self.encoder_end = 1
        self.decoder = []

    def add(self, layer, input_dim=()):
        from ..layers import DenseLayer, HighwayLayer, LSTM, RLayer

        if type(layer) not in (DenseLayer, HighwayLayer, RLayer, LSTM):
            raise NotImplementedError(str(layer), "not yet implemented in autoencoder!")
        Network.add(self, layer, input_dim)
        self.encoder_end += 1

        if self.decoder_type == "learnable":
            caps = layer.capsule
            self.decoder = type(layer).from_capsule(caps)

    def pop(self):
        self.layers = self.layers[:self.encoder_end-1]
        self.encoder_end -= 1
        self._finalized = False

    def finalize(self, cost, optimizer="sgd"):
        from ..costs import cost_fns
        from ..optimizers import optimizers
        from ..layers import DenseLayer

        for layer in reversed(self.layers[1:]):
            decoder_layer = type(layer)
            layer.connect(self, self.layers[-1].outshape)
            self.layers.append(layer)
            self.architecture.append(str(layer))
        self.layers.append(DenseLayer(self.layers[0].neurons))
        self.layers[-1].connect(self, self.layers[-2].outshape)
        self.architecture.append(str(self.layers[-1]))
        for layer in self.layers:
            if layer.trainable:
                if isinstance(optimizer, str):
                    optimizer = optimizers[optimizer](layer)
                layer.optimizer = optimizer
        self.cost = cost_fns[cost] if isinstance(cost, str) else cost
        self._finalized = True

    def encode(self, X):
        for layer in self.layers[:self.encoder_end]:
            X = layer.feedforward(X)
        return X

    def decode(self, X):
        for layer in self.layers[self.encoder_end:]:
            X = layer.feedforward(X)
        return X

    # noinspection PyMethodOverriding
    def fit(self, X, batch_size=20, epochs=10, monitor=(), validation=(), verbose=1, shuffle=True):
        Network.fit(self, X, X, batch_size, epochs, monitor, validation, verbose, shuffle)

    def fit_csxdata(self, frame, batch_size=20, epochs=10, monitor=(), verbose=1, shuffle=True):
        X, _ = frame.table("learning")
        if frame.n_testing:
            vX, _ = frame.table("testing")
        else:
            vX = None
        Network.fit(self, X, X, batch_size, epochs, monitor, (vX, vX), verbose, shuffle)

    # noinspection PyMethodOverriding
    def gradient_check(self, X, verbose=1, epsilon=1e-5):
        return Network.gradient_check(self, X, X, verbose, epsilon)


def _xent_hackaround(networkobj, xentobj):
    xentobj.__call__ = [xentobj.call_on_sigmoid, xentobj.call_on_softmax][int(networkobj.layers[-1].activation.type == "softmax")]
    return xentobj
