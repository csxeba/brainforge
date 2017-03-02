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
        capsule = {
            "name": self.name,
            "cost": self.cost,
            "optimizers": self.optimizer,
            "architectures": self.architecture[:],
            "layers": [layer.capsule() for layer in self.layers]}

        if dumppath is None:
            return capsule
        else:
            import pickle
            with open(dumppath, "wb") as outfl:
                pickle.dump(capsule, outfl)
                outfl.close()

    @classmethod
    def from_capsule(cls, capsule):

        def prepare_capsule(caps):
            if isinstance(caps, str):
                import pickle
                infl = open(caps, "rb")
                caps = pickle.load(infl)
                infl.close()
            return caps

        from brainforge.util.shame import translate_architecture as trsl
        from brainforge.optimizers import optimizer as opts

        c = prepare_capsule(capsule)

        net = Network(input_shape=c["layers"][0][0], name=c["name"])

        for layer_name, layer_capsule in zip(c["architectures"], c["layers"]):
            if layer_name[:5] == "Input":
                continue
            layer_cls = trsl(layer_name)
            layer = layer_cls.from_capsule(layer_capsule)
            net.add(layer)

        opti = c["optimizers"]
        if isinstance(opti, str):
            opti = opts[opti]()
        net.finalize(cost=c["cost"], optimizer=opti)

        for layer, lcaps in zip(net.layers, c["layers"]):
            if layer.weights is not None:
                layer.set_weights(lcaps[-1], fold=False)

        return net

    # ---- Methods for architectures building ----

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

    def finalize(self, cost, optimizer="sgd"):
        from ..costs import cost_fns as costs
        from ..optimizers import optimizer as opt

        self.cost = costs[cost] if isinstance(cost, str) else cost
        self.optimizer = optimizer

        if str(cost).lower() == "xent":
            if str(self.layers[-1].activation) not in ("softmax", "sigmoid"):
                errmsg = "Sorry, xent only supported with softmax or sigmoid output activation!\n"
                errmsg += "{} not supported!".format(str(self.layers[-1].activation))
                raise RuntimeError(errmsg)
            self.layers[-1].activation.derivative = lambda X: np.ones_like(X)

            # TODO: discard this ugly hack!
            if str(self.layers[-1].activation) == "sigmoid":
                from ..costs._costs import _XentOnSigmoid
                self.cost = _XentOnSigmoid()
            else:
                from ..costs._costs import _XentOnSoftmax
                self.cost = _XentOnSoftmax()

        for layer in self.layers:
            if layer.trainable:
                if isinstance(optimizer, str):
                    optimizer = opt[optimizer](layer)
                layer.optimizer = optimizer

        self._finalized = True

    def pop(self):
        self.layers.pop()
        self.architecture.pop()
        self._finalized = False

    # ---- Methods for model fitting ----

    def fit(self, X, Y, batch_size=20, epochs=30, monitor=(), validation=(), verbose=1, shuffle=True):

        if not self._finalized:
            raise RuntimeError("Architecture not finalized!")

        costs = []
        for epoch in range(1, epochs+1):
            if shuffle:
                arg = np.arange(X.shape[0])
                np.random.shuffle(arg)
                X, Y = X[arg], Y[arg]
            if verbose:
                print("Epoch {}/{}".format(epoch, epochs))
            costs += self.epoch(X, Y, batch_size, monitor, validation, verbose)
        return costs

    def fit_csxdata(self, frame, batch_size=20, epochs=10, monitor=(), verbose=1, shuffle=True):
        fanin, outshape = frame.neurons_required
        if fanin != self.layers[0].outshape or outshape != self.layers[-1].outshape:
            errstring = "Network configuration incompatible with supplied dataframe!\n"
            errstring += "fanin: {} <-> InputLayer: {}\n".format(fanin, self.layers[0].outshape)
            errstring += "outshape: {} <-> Net outshape: {}\n".format(outshape, self.layers[-1].outshape)
            raise RuntimeError(errstring)

        X, Y = frame.table("learning")
        validation = frame.table("testing")

        return self.fit(X, Y, batch_size, epochs, monitor, validation, verbose, shuffle)

    def epoch(self, X, Y, batch_size, monitor, validation, verbose):

        self.learning = True

        self.N = X.shape[0]

        def print_progress():
            classificaton = "acc" in monitor
            results = self.evaluate(*validation, classify=classificaton)

            chain = "testing cost: {0:.5f}"
            if classificaton:
                tcost, tacc = results
                accchain = "\taccuracy: {0:.2%}".format(tacc)
            else:
                tcost = results[0]
                accchain = ""
            print(chain.format(tcost) + accchain, end="")

        costs = []
        batches = ((X[start:start+batch_size], Y[start:start+batch_size])
                   for start in range(0, self.N, batch_size))

        for bno, (inputs, targets) in enumerate(batches):
            costs.append(self._fit_batch(inputs, targets))
            if verbose:
                done = ((bno * batch_size) + self.m) / self.N
                print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(done, np.mean(costs)), end="")

        if verbose and validation:
            print_progress()
            print()
        elif verbose:
            print()
        self.age += 1
        self.learning = False
        return costs

    def _fit_batch(self, X, Y, parameter_update=True):
        self.prediction(X)
        self.backpropagation(Y)
        if parameter_update:
            self._parameter_update()

        return self.cost(self.output, Y) / self.m

    def backpropagation(self, Y):
        error = self.cost.derivative(self.layers[-1].output, Y)
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)

    def _parameter_update(self):
        for layer in filter(lambda x: x.trainable, self.layers):
            layer.optimizer(self.m)

    # ---- Methods for forward propagation ----

    def regress(self, X):
        return self.prediction(X)

    def classify(self, X):
        return np.argmax(self.prediction(X), axis=1)

    def prediction(self, X):
        self.m = X.shape[0]
        for layer in self.layers:
            X = layer.feedforward(X)
        return X

    def predict_proba(self, X):
        return self.prediction(X)

    def evaluate(self, X, Y, classify=True):
        predictions = self.prediction(X)
        cost = self.cost(predictions, Y) / Y.shape[0]
        if classify:
            pred_classes = np.argmax(predictions, axis=1)
            trgt_classes = np.argmax(Y, axis=1)
            eq = np.equal(pred_classes, trgt_classes)
            acc = np.mean(eq)
            return cost, acc
        return cost

    # ---- Some utilities ----

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
        from ..costs import cost_fns as costs
        from ..optimizers import optimizer as opt
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
                    optimizer = opt[optimizer](layer)
                layer.optimizer = optimizer
        self.cost = costs[cost] if isinstance(cost, str) else cost
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
