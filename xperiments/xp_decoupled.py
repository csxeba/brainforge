from collections import deque

from csxdata.utilities.loader import pull_mnist_data

from brainforge import BackpropNetwork
from brainforge.layers.abstract_layer import LayerBase, NoParamMixin
from brainforge.layers import DenseLayer


class DNI(NoParamMixin, LayerBase):

    def __init__(self, synth: BackpropNetwork=None, **kw):
        super().__init__(**kw)
        self.synth = synth
        self.memory = deque()
        self._predictor = None
        self._previous = None

    def _default_synth(self):
        synth = BackpropNetwork(input_shape=self.inshape, layerstack=[
            DenseLayer(self.inshape[0], activation="tanh"),
            DenseLayer(self.inshape[0], activation="linear"),
        ], cost="mse", optimizer="sgd")
        return synth

    def connect(self, to, inshape):
        super().connect(to, inshape)
        self._previous = to.layers[-1]
        if self.synth is None:
            self.synth = self._default_synth()

    def feedforward(self, X):
        delta = self.synth.predict(X)
        self._previous.backpropagate(delta)
        if self.brain.learning:
            self.memory.append(delta)
        return X

    def backpropagate(self, delta):
        m = self.memory.popleft()
        print(f"\rSynth cost: {self.synth.cost(m, delta).sum():.4f}", end="")
        self.synth.learn_batch(m, delta)

    @property
    def outshape(self):
        return self.inshape

    @classmethod
    def from_capsule(cls, capsule):
        pass

    def __str__(self):
        return "DNI"


def build_decoupled_net(inshape, outshape):
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        DenseLayer(60, activation="tanh"), DNI(),
        DenseLayer(outshape, activation="softmax")
    ], cost="xent", optimizer="adam")
    return net


def build_normal_net(inshape, outshape):
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        DenseLayer(60, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ], cost="xent", optimizer="adam")
    return net


def xperiment():
    lX, lY, tX, tY = pull_mnist_data()
    net = build_decoupled_net(lX.shape[1:], lY.shape[1:])
    for epoch in range(30):
        net.fit(lX, lY, batch_size=128, epochs=1, verbose=0)
        cost, acc = net.evaluate(tX, tY)
        print(f"\nEpoch {epoch} done! Network accuracy: {acc:.2%}")


if __name__ == '__main__':
    xperiment()
