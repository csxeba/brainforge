from brainforge import BackpropNetwork
from brainforge.layers import ConvLayer, PoolLayer, Flatten, DenseLayer, Activation
from brainforge.optimization import RMSprop

from csxdata.utilities.loader import pull_mnist_data


lX, lY, tX, tY = pull_mnist_data(fold=True)
ins, ous = lX.shape[1:], lY.shape[1:]
net = BackpropNetwork(input_shape=ins, layerstack=[
    ConvLayer(3, 8, 8, compiled=False),
    PoolLayer(3, compiled=False), Activation("tanh"),
    Flatten(), DenseLayer(60, activation="tanh"),
    DenseLayer(ous[0], activation="softmax")
], cost="cxent", optimizer=RMSprop(eta=0.01))

net.fit(lX, lY, batch_size=32, epochs=10, validation=(tX, tY))
