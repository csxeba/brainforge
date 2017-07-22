from csxdata import roots, CData

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer
from brainforge.optimization import SGD
from brainforge.gradientcheck import GradientCheck

mnist = CData(roots["misc"] + "mnist.pkl.gz", cross_val=10000, fold=False)
inshape, outshape = mnist.neurons_required


network = BackpropNetwork(input_shape=inshape, layerstack=[
    DenseLayer(30, activation="sigmoid"),
    DenseLayer(outshape, activation="softmax")
], cost="xent", optimizer=SGD(eta=3.))

gcsuite = GradientCheck(network)
gcsuite.run(*mnist.table("testing", m=5))

network.fit(*mnist.table("learning"), validation=mnist.table("testing"))
