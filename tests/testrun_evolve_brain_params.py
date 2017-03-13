from csxdata import CData, roots

from brainforge import Network
from brainforge.layers import DenseLayer, DropOut
from brainforge.optimizers import Evolution

data = CData(roots["misc"] + "mnist.pkl.gz", cross_val=10000, fold=False)
net = Network.from_csxdata(data, layers=(
    DenseLayer(60, activation="sigmoid"),
    DenseLayer(data.neurons_required[1], activation="softmax")
))
net.finalize("mse", optimizer=Evolution(mate_function=lambda ind1, ind2: (ind1 + ind2) / 2.))

net.fit_csxdata(data, 1000, monitor=["acc"])
