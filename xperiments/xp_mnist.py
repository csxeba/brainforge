from csxdata.utilities.loader import pull_mnist_data

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer


ann = BackpropNetwork(input_shape=784, layerstack=[
    DenseLayer(60, activation="tanh"),
    DenseLayer(10, activation="softmax")
])

lX, lY, tX, tY = pull_mnist_data()

ann.fit(lX, lY, 32, 10, validation=(tX, tY))
