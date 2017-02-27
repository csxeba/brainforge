import time

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from brainforge import Network
from brainforge.layers import DenseLayer, Flatten
from brainforge.evolution import Population, to_phenotype

from csxdata import CData, roots

matplotlib.use("Qt5Agg")

dataroot = roots["misc"] + "mnist.pkl.gz"
frame = CData(dataroot, headers=None)
frame.transformation = "std"
tX, tY = frame.table("learning", shuff=True, m=10000)

inshape, outshape = frame.neurons_required

# Genome will be the number of hidden neurons
# at each network layer.
ranges = ((2, 100), (2, 100))


def phenotype_to_ann(phenotype):
    net = Network(inshape, layers=[
        Flatten(),
        DenseLayer(int(phenotype[0]), activation="tanh"),
        DenseLayer(int(phenotype[1]), activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ])
    net.finalize(cost="xent", optimizer="adagrad")
    return net


# Define the fitness function -> evaluate the neural network
def fitness(genotype):
    start = time.time()
    phenotype = to_phenotype(genotype, ranges)
    net = phenotype_to_ann(phenotype)
    net.fit(tX, tY, batch_size=50, epochs=10, verbose=0)
    score = net.evaluate(*frame.table("testing", shuff=True, m=100))[-1]
    timereq = time.time() - start
    return (1. - score), timereq  # fitness is minimized, so we need error rate


pop = Population(
    loci=len(ranges),
    limit=21,
    fitness_function=fitness,
    fitness_weights=[10., 1.])

means, totals, bests = pop.run(epochs=30, verbosity=4,
                               survival_rate=0.8,
                               mutation_rate=0.1)
logs = pop.run(epochs=3, verbosity=1,
               survival_rate=0.66,
               mutation_rate=0.0)

means += logs[0]
totals += logs[1]
bests += logs[2]

print("\nThe winner is:")
winner = phenotype_to_ann(to_phenotype(pop.best, ranges))
winner.describe(1)
winner.fit_csxdata(frame, epochs=30, monitor=["acc"])

Xs = np.arange(1, len(means)+1)
fig, axes = plt.subplots(2, sharex=True)
axes[0].plot(Xs, totals)
axes[0].set_title("Total grade")
axes[1].plot(Xs, means, color="blue")
axes[1].plot(Xs, bests, color="red")
axes[1].set_title("Means and bests")

plt.show()
