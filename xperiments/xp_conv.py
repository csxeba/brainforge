from verres.data import MNIST

from brainforge.learner import BackpropNetwork
from brainforge.layers import ConvLayer, PoolLayer, Flatten, DenseLayer, Activation
from brainforge.optimizers import RMSprop
from brainforge.gradientcheck import GradientCheck

X, Y = MNIST().table("train")
X = X[..., 0][:, None, ...]
ins, ous = X.shape[1:], Y.shape[1:]
net = BackpropNetwork(input_shape=ins, layerstack=[
    ConvLayer(32, 3, 3, compiled=1),
    Activation("tanh"),
    ConvLayer(64, 3, 3, compiled=1),
    PoolLayer(2, compiled=1),
    Activation("tanh"),
    Flatten(),
    DenseLayer(ous[0], activation="softmax")
], cost="cxent", optimizer="adam")

net.learn_batch(X[-5:], Y[-5:])
net.age += 1

# GradientCheck(net, epsilon=1e-5).run(X[:5], Y[:5], throw=True)

net.fit(X, Y, batch_size=32, epochs=10, metrics=["acc"])