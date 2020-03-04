from verres.data import MNIST

from brainforge.learner import BackpropNetwork
from brainforge.layers import ConvLayer, PoolLayer, Flatten, Dense, Activation
from brainforge import gradientcheck

X, Y = MNIST().table("train")
X = X[..., 0][:, None, ...]
ins, ous = X.shape[1:], Y.shape[1:]
net = BackpropNetwork(input_shape=ins, layerstack=[
    ConvLayer(32, 3, 3, compiled=1),
    Activation("relu"),
    ConvLayer(64, 3, 3, compiled=1),
    PoolLayer(2, compiled=1),
    Activation("relu"),
    Flatten(),
    Dense(ous[0], activation="softmax")
], cost="cxent", optimizer="adam")

gradientcheck.run(net, X[:5], Y[:5], epsilon=1e-5, throw=True)

net.fit(X, Y, batch_size=32, epochs=10, metrics=["acc"])
