from verres.data import inmemory

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer, Flatten

mnist = inmemory.MNIST()
(lX, lY), (tX, tY) = mnist.table("train", shuffle=True), mnist.table("val", shuffle=False)

ann = BackpropNetwork(input_shape=lX.shape[1:], layerstack=[
    Flatten(),
    DenseLayer(64, activation="tanh"),
    DenseLayer(10, activation="softmax")
], cost="cxent", optimizer="adam")

ann.fit(lX, lY, 32, 10, metrics=["acc"], validation=(tX, tY))
