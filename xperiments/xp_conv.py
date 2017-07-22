from csxdata import CData, roots

from brainforge import BackpropNetwork
from brainforge.layers import ConvLayer, PoolLayer, Flatten, DenseLayer, Activation
from brainforge.optimization import RMSprop

data = CData(roots["misc"] + "mnist.pkl.gz", cross_val=10000, fold=True)
ins, ous = data.neurons_required
net = BackpropNetwork(input_shape=ins, layerstack=[
    ConvLayer(3, 8, 8, compiled=False),
    PoolLayer(3, compiled=False), Activation("tanh"),
    Flatten(), DenseLayer(60, activation="tanh"),
    DenseLayer(ous, activation="softmax")
], cost="xent", optimizer=RMSprop(eta=0.01))

net.fit_generator(data.batchgen(bsize=20, infinite=True), lessons_per_epoch=60000, epochs=30,
                  validation=data.table("testing"))
