from csxdata import roots, Sequence

from brainforge import BackpropNetwork
from brainforge.layers import LSTM, DenseLayer
from brainforge.optimization import RMSprop


data = Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=5, floatX="float64")
inshape, outshape = data.neurons_required
net = BackpropNetwork(input_shape=inshape, layerstack=[
    LSTM(60, activation="tanh"),
    DenseLayer(60, activation="tanh"),
    DenseLayer(outshape, activation="softmax")
], cost="xent", optimizer=RMSprop(eta=0.01))

net.fit(*data.table("learning"), validation=data.table("testing"))
