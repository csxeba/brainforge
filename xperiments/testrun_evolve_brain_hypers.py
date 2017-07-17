import time

import numpy as np
from matplotlib import pyplot as plt

from brainforge import BackpropNetwork
from brainforge.architecture import DenseLayer, DropOut
from brainforge.evolution import Population, to_phenotype

from csxdata import CData, roots

frame = CData(roots["misc"] + "mnist.pkl.gz", fold=False)

inshape, outshape = frame.neurons_required

# Genome will be the number of hidden neurons at two network DenseLayers.
ranges = ((10, 300), (0, 0.75), (10, 300), (0, 0.75))
# We determine 2 fitness values: the network's classification error and
# the time required to run the net. These two values will both be minimized
# and the accuracy will be considered with a 20x higher weight.
fweights = (1, 1)


def phenotype_to_ann(phenotype):
    net = BackpropNetwork(inshape, layers=[
        DenseLayer(int(phenotype[0]), activation="tanh"),
        DropOut(dropchance=phenotype[1]),
        DenseLayer(int(phenotype[2]), activation="tanh"),
        DropOut(dropchance=phenotype[3]),
        DenseLayer(outshape, activation="softmax")
    ])
    net.finalize(cost="xent", optimizer="momentum")
    return net


# Define the fitness function
def fitness(genotype):
    start = time.time()
    net = phenotype_to_ann(to_phenotype(genotype, ranges))
    net.fit(*frame.table("learning", m=1000), batch_size=50, epochs=1, verbose=0)
    score = net.evaluate(*frame.table("testing", m=10), classify=True)[-1]
    error_rate = 1. - score
    time_req = time.time() - start
    return error_rate, time_req


# Build a population of 12 individuals. grade_function and mate_function are
# left to defaults.
pop = Population(loci=4, limit=15,
                 fitness_function=fitness,
                 fitness_weights=fweights)
# The population is optimized for 12 rounds with the hyperparameters below.
# at every 3 rounds, we force a complete-reupdate of fitnesses, because the
# neural networks utilize randomness due to initialization, random batches, etc.
means, stds, bests = pop.run(epochs=30,
                             survival_rate=0.3,
                             mutation_rate=0.05,
                             force_update_at_every=3)

Xs = np.arange(1, len(means)+1)
plt.title("Population grade dynamics of\nevolutionary hyperparameter optimization")
plt.plot(Xs, means, color="blue")
plt.plot(Xs, means+stds, color="green", linestyle="--")
plt.plot(Xs, means-stds, color="green", linestyle="--")
plt.plot(Xs, bests, color="red")
plt.show()
