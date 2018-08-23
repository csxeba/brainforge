from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer
from brainforge.gradientcheck import GradientCheck
from brainforge.util import etalon

X, Y = etalon
inshape, outshape = X.shape[1:], Y.shape[1:]

network = BackpropNetwork(input_shape=inshape, layerstack=[
    DenseLayer(64, activation="tanh"),
    DenseLayer(32, activation="tanh"),
    DenseLayer(outshape, activation="sigmoid", trainable=False)
], cost="mse", optimizer="sgd")
network.fit(X[5:], Y[5:], epochs=1, batch_size=len(X)-5, verbose=0)

gcsuite = GradientCheck(network, epsilon=1e-3)
gcsuite.run(X[:5], Y[:5])
