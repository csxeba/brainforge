from csxdata import Sequence, roots

from brainforge import BackpropNetwork
from brainforge.layers import RLayer, DenseLayer
from brainforge.optimization import RMSprop
from brainforge.gradientcheck import GradientCheck

data = Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=6, floatX=float)
inshape, outshape = data.neurons_required
net = BackpropNetwork(input_shape=inshape, layerstack=[
    RLayer(10, activation="tanh", compiled=True),
    DenseLayer(30, activation="tanh", compiled=True),
    DenseLayer(outshape, activation="softmax", compiled=True)
], cost="xent", optimizer=RMSprop(eta=0.01))

GradientCheck(net).run(*data.table("testing", m=10), throw=True)

net.fit(*data.table("learning"), validation=data.table("testing"))
