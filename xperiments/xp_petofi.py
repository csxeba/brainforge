from csxdata import roots, Sequence

from brainforge import BackpropNetwork
from brainforge.layers import RLayer, DenseLayer
from brainforge.optimization import RMSprop
from brainforge.gradientcheck import GradientCheck


data = Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=5, floatX="float64")
inshape, outshape = data.neurons_required
net = BackpropNetwork(input_shape=inshape, layerstack=[
    RLayer(30, activation="tanh", compiled=False),
    DenseLayer(30, activation="tanh"),
    DenseLayer(outshape, activation="softmax")
], cost="xent", optimizer=RMSprop(eta=0.01))

# net.fit(*data.table("learning", m=5), epochs=1, verbose=0)

gcsuite = GradientCheck(net)
gcsuite.run(*data.table("testing", m=5))

net.fit(*data.table("learning"))
