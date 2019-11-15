from brainforge.learner import BackpropNetwork
from brainforge.layers import Dense
from brainforge.optimizers import Momentum
from brainforge.util import etalon


class DNI:

    def __init__(self, bpropnet, synth):
        self.bpropnet = bpropnet
        self.synth = synth
        self._predictor = None

    def predictor_coro(self):
        prediction = None
        delta_backwards = None
        while 1:
            inputs = yield prediction, delta_backwards
            prediction = self.bpropnet.predict(inputs)
            synthesized_delta = self.synth.predict(prediction)
            self.bpropnet.update(len(inputs))
            delta_backwards = self.bpropnet.backpropagate(synthesized_delta)

    def predict(self, X):
        if self._predictor is None:
            self._predictor = self.predictor_coro()
            next(self._predictor)
        return self._predictor.send(X)

    def udpate(self, true_delta):
        synthesizer_delta = self.synth.cost.derivative(
            self.synth.output, true_delta
        )
        self.synth.backpropagate(synthesizer_delta)
        self.synth.update(len(true_delta))


def build_net(inshape, outshape):
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        Dense(30, activation="tanh"),
        Dense(outshape, activation="softmax")
    ], cost="cxent", optimizer=Momentum(0.01))
    return net


def build_synth(inshape, outshape):
    synth = BackpropNetwork(input_shape=inshape, layerstack=[
        Dense(outshape)
    ], cost="mse", optimizer=Momentum(0.01))
    return synth


X, Y = etalon

predictor = build_net(X.shape[1:], Y.shape[1:])
pred_os = predictor.layers[-1].outshape
synthesizer = build_synth(pred_os, pred_os)
dni = DNI(predictor, synthesizer)

pred, delta = dni.predict(X)
