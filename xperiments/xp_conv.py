from csxdata import CData, roots

from brainforge import BackpropNetwork
from brainforge.layers import ConvLayer, PoolLayer, Flatten, DenseLayer, Activation
from brainforge.optimization import RMSprop

data = CData(roots["misc"] + "mnist.pkl.gz", cross_val=10000, fold=True, floatX="float64")
ins, ous = data.neurons_required
net = BackpropNetwork(input_shape=ins, layerstack=[
    ConvLayer(3, 8, 8, compiled=True),
    PoolLayer(3, compiled=True), Activation("tanh"),
    Flatten(), DenseLayer(60, activation="tanh"),
    DenseLayer(ous, activation="softmax")
], cost="xent", optimizer=RMSprop(eta=0.01))

net.fit_generator(data.batchgen(bsize=20, infinite=True), lessons_per_epoch=1000, epochs=10,
                  validation=data.table("testing", m=1000))
