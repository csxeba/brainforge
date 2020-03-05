from verres.data import inmemory

from brainforge import Backpropagation
from brainforge.layers import Dense, Flatten

mnist = inmemory.MNIST()
(lX, lY), (tX, tY) = mnist.table("train", shuffle=True), mnist.table("val", shuffle=False)

ann = Backpropagation(input_shape=lX.shape[1:], layerstack=[
    Flatten(),
    Dense(64, activation="tanh"),
    Dense(10, activation="softmax")
], cost="cxent", optimizer="adam")

ann.fit(lX, lY, 32, 10, metrics=["acc"], validation=(tX, tY))
