from csxdata import CData, roots

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.evolution import Population
from brainforge.optimizers import Evolution

data = CData(roots["misc"] + "mnist.pkl.gz", cross_val=10000, fold=False,
             floatX="float64")
net = Network.from_csxdata(data, layers=(
    DenseLayer(60, activation="sigmoid"),
    DenseLayer(data.neurons_required[1], activation="softmax")
))
pop = Population(net.nparams, fitness_function=Evolution.default_fitness)
net.finalize("xent", optimizer=Evolution(pop, evolution_epochs=3))
net.fit_csxdata(data, 1000, monitor=["acc"])
