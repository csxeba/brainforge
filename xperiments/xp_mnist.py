from verres.data import load_mnist

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer, ConvLayer, Activation, PoolLayer, Flatten


lX, lY, tX, tY = load_mnist()
lX = lX.transpose((0, 3, 1, 2))
tX = tX.transpose((0, 3, 1, 2))

ann = BackpropNetwork(input_shape=lX.shape[1:], layerstack=[
    ConvLayer(8), Activation("relu"),  # 30
    ConvLayer(8), PoolLayer(2), Activation("relu"),  # 28->14
    ConvLayer(16), Activation("relu"),  # 12
    ConvLayer(16), PoolLayer(2), Activation("relu"),  # 10->5
    ConvLayer(32), Activation("relu"),  # 3
    Flatten(),
    DenseLayer(32, activation="relu"),
    DenseLayer(10, activation="softmax")
], cost="cxent", optimizer="adam")

ann.fit(lX, lY, 32, 10, validation=(tX, tY))
