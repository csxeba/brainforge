import numpy as np

from brainforge.learner import BackpropNetwork
from brainforge.layers import ConvLayer, PoolLayer, Flatten, DenseLayer, Activation
from brainforge.optimization import RMSprop
from brainforge.gradientcheck import GradientCheck

X, Y = np.random.randn(5, 3, 12, 12), np.ones((5, 2))
ins, ous = X.shape[1:], Y.shape[1:]
net = BackpropNetwork(input_shape=ins, layerstack=[
    ConvLayer(3, 3, 3, compiled=1),
    Activation("tanh"),
    ConvLayer(3, 3, 3, compiled=1),
    PoolLayer(2, compiled=1),
    Activation("tanh"),
    Flatten(),
    DenseLayer(ous[0], activation="linear", trainable=False)
], cost="mse", optimizer=RMSprop(eta=0.01))

net.learn_batch(X, Y)
net.age += 1

GradientCheck(net, epsilon=1e-5).run(X, Y, throw=True)
