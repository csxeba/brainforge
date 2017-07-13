from csxdata import CData, roots

from brainforge import GradientLearner
from brainforge.architecture import DenseLayer
from brainforge.evolution import Population


def meanmate(ind1, ind2):
    return (ind1 + ind2) / 2.


def fitness(W, *args, **kw):
    net.set_weights(W, fold=True)
    cost, acc = net.evaluate(*data.table("learning", m=10000))
    return cost,


data = CData(roots["misc"] + "mnist.pkl.gz", cross_val=10000, fold=False,
             floatX="float64")
net = GradientLearner(data.neurons_required[0], layers=(
    DenseLayer(60, activation="sigmoid"),
    DenseLayer(data.neurons_required[1], activation="softmax")
))
pop = Population(net.nparams, fitness_function=fitness)
net.finalize("xent")

print("Initial acc:", net.evaluate(*data.table("testing"))[1])

for epoch in range(10):
    print("Epoch", epoch, end=" ")
    pop.run(3, verbosity=1, survival_rate=0.2, mutation_rate=0.01)
    net.set_weights(pop.best, fold=True)
    print("Current acc:", net.evaluate(*data.table("testing"))[1])
