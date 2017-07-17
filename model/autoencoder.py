from .backpropagation import BackpropNetwork


class Autoencoder(BackpropNetwork):

    def __init__(self, inshape=(), decoder_type="learnable", name=""):
        super().__init__(inshape, name=name)
        if decoder_type not in ("learnable", "fixed", "built", "single", None):
            raise ValueError('decoder_type should be one of:\n"learnable", "mirrored", "built", "single", None')
        self.decoder_type = decoder_type
        self.encoder_end = 1
        self.decoder = []

    def add(self, layer):
        from ..architecture import DenseLayer, HighwayLayer, LSTM, RLayer

        if type(layer) not in (DenseLayer, HighwayLayer, RLayer, LSTM):
            raise NotImplementedError(str(layer), "not yet implemented in autoencoder!")
        GradientLearner.add(self, layer)
        self.encoder_end += 1

        if self.decoder_type == "learnable":
            caps = layer.capsule
            self.decoder = type(layer).from_capsule(caps)

    def pop(self):
        self.layers = self.layers[:self.encoder_end-1]
        self.encoder_end -= 1
        self._finalized = False

    def finalize(self, cost="mse", optimizer="sgd"):
        from ..cost import cost_functions
        from ..optimization import optimizers
        from ..architecture import DenseLayer

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
        self.cost = cost_functions[cost] if isinstance(cost, str) else cost
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
        super().fit(X, X, batch_size, epochs, monitor, validation, verbose, shuffle)

    # noinspection PyMethodOverriding
    def gradient_check(self, X, verbose=1, epsilon=1e-5):
        return GradientLearner.gradient_check(self, X, X, verbose, epsilon)