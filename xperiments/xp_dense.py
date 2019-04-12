from brainforge.model import BackpropNetwork
from brainforge.layers import Linear, Activation
from brainforge.gradientcheck import GradientCheck
from brainforge.util import etalon

X, Y = etalon
inshape, outshape = X.shape[1:], Y.shape[1:]

network = BackpropNetwork(input_shape=inshape, layerstack=[
    Linear(10), Activation("sigmoid"),
    Linear(outshape)
], cost="xent", optimizer="sgd")
network.fit(X[5:], Y[5:], epochs=1, batch_size=len(X)-5, verbose=0)

gcsuite = GradientCheck(network, epsilon=1e-3)
gcsuite.run(X[:5], Y[:5])
