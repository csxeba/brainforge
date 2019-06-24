import numpy as np
from verres.data import generators

from brainforge import LayerStack, BackpropNetwork
from brainforge.layers.abstract_layer import LayerBase, NoParamMixin
from brainforge.layers import DenseLayer
from brainforge.util.typing import white_like


class Sampler(NoParamMixin, LayerBase):

    def __init__(self):
        super().__init__()
        self._outshape = None
        self.epsilon = None

    def connect(self, brain):
        self._outshape = brain.outshape
        super().connect(brain)

    def feedforward(self, X):
        """O = M + V * E"""
        self.inputs = X.copy()
        self.epsilon = white_like(X[0])
        return X[0] + X[1] * self.epsilon

    @property
    def outshape(self):
        return self._outshape

    def backpropagate(self, delta):
        return np.stack([delta, delta * self.epsilon], axis=0)


Z = 2

layers = LayerStack(784, [
    DenseLayer(60, activation="relu"),
    DenseLayer(30, activation="relu")
])
encoder = BackpropNetwork(layers, cost="mse", optimizer="momentum")

mean_z = BackpropNetwork(input_shape=30, layerstack=[DenseLayer(Z)], cost="mse", optimizer="momentum")
stdev_z = BackpropNetwork(input_shape=30, layerstack=[DenseLayer(Z)], cost="mse", optimizer="momentum")

sampler = BackpropNetwork(input_shape=Z, layerstack=[Sampler()], cost="mse", optimizer="momentum")

decoder = BackpropNetwork(input_shape=Z, layerstack=[DenseLayer(30, activation="relu"),
                                                     DenseLayer(60, activation="relu"),
                                                     DenseLayer(784, activation="linear")],
                          cost="mse", optimizer="momentum")


for epoch in range(10):
    print("\n\nEpoch", epoch)
    for i, (x, y) in enumerate(ds.batch_stream(batchsize=32)):
        m = len(x)
        enc = encoder.predict(x)
        mean = mean_z.predict(enc)
        stdev = stdev_z.predict(enc)
        smpl = sampler.predict(np.stack([mean, stdev], axis=0))
        dcd = decoder.predict(smpl)

        delta = decoder.cost.derivative(dcd, x)
        delta = decoder.backpropagate(delta)
        d_mean, d_std = sampler.backpropagate(delta)
        delta = mean_z.backpropagate(d_mean) + stdev_z.backpropagate(d_std)
        encoder.backpropagate(delta)

        encoder.update(m)
        mean_z.update(m)
        stdev_z.update(m)
        decoder.update(m)

        print("\rDone: {:.2%} Cost: {:4f}".format(i*32 / len(ds), decoder.cost(dcd, x)), end="")
