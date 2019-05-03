import numpy as np

from brainforge.learner import BackpropNetwork
from brainforge.layers import LSTM, DenseLayer
from brainforge.gradientcheck import GradientCheck

np.random.seed(1337)

DSHAPE = 10, 1, 15
OUTSHP = 10, 15
X = np.random.randn(*DSHAPE)
Y = np.random.randn(*OUTSHP)

net = BackpropNetwork(input_shape=DSHAPE[1:], layerstack=[
    LSTM(32, activation="tanh"),
    DenseLayer(10, activation="tanh", trainable=False),
    DenseLayer(OUTSHP[1:], activation="linear", trainable=False)
], cost="mse", optimizer="sgd")

net.fit(X, Y, epochs=1, verbose=0)
GradientCheck(net, epsilon=1e-6, display=True).run(X, Y, throw=True)
