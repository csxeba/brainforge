from csxdata import Sequence, roots

from brainforge import BackpropNetwork
from brainforge.layers import LSTM as RArch, DenseLayer
from brainforge.optimization import RMSprop
from brainforge.gradientcheck import GradientCheck

data = Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=3, floatX=float)
inshape, outshape = data.neurons_required
net = BackpropNetwork(input_shape=inshape, layerstack=[
    RArch(10, activation="tanh", compiled=False),
    DenseLayer(outshape, activation="softmax")
], cost="xent", optimizer=RMSprop(eta=0.01))

net.fit(*data.table("learning", m=5), verbose=0, epochs=1)
GradientCheck(net).run(*data.table("testing", m=10), throw=True)
