import numpy as np

from brainforge.learner import Backpropagation
from brainforge.layers import LSTM, Dense
from brainforge.gradientcheck import GradientCheck

# np.random.seed(1337)

DSHAPE = 20, 10, 1
OUTSHP = 20, 1
X = np.random.randn(*DSHAPE)
Y = np.random.randn(*OUTSHP)

net = Backpropagation(input_shape=DSHAPE[1:], layerstack=[
    LSTM(16, activation="tanh", compiled=1),
    Dense(OUTSHP[1:], activation="linear", trainable=0)
], cost="mse", optimizer="sgd")

net.fit(X, Y, epochs=1, verbose=0)
GradientCheck(net, display=True).run(X, Y, throw=True)
