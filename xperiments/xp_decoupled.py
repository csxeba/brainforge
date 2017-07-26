from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer
from brainforge.optimization import Momentum
from brainforge.util import etalon


class DNI:

    """
    Decoupled neural interface, implemented with coroutines
    """

    def __init__(self, bpropnet, synth):
        self.bpropnet = bpropnet
        self.synth = synth

    def predict(self, X):
        while 1:
            pred = self.bpropnet.predict(X)
            syn_delta = self.synth.predict(pred)
            self.bpropnet.update(len(X))
            X = yield pred, self.bpropnet.backpropagate(syn_delta)

    def udpate(self, delta):
        while 1:
            sdelta = self.synth.cost.derivative(
                self.synth.output, delta
            )
            self.synth.backpropagate(sdelta)
            self.synth.update(len(delta))


def build_net(inshape, outshape):
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        DenseLayer(30, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ], cost="xent", optimizer=Momentum(0.01))
    return net


def build_synth(inshape):
    synth = BackpropNetwork(input_shape=inshape, layerstack=[
        DenseLayer(inshape)
    ], cost="mse", optimizer=Momentum(0.01))
    return synth


dni1 = DNI(build_net)