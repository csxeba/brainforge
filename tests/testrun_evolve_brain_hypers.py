import time

import numpy as np

from matplotlib import pyplot as plt

from brainforge import Network
from brainforge.layers import DenseLayer, Flatten
from brainforge.evolution import Population, to_phenotype

from csxdata import CData, roots

dataroot = roots["misc"] + "mnist.pkl.gz"
frame = CData(dataroot, headers=None)
frame.transformation = "std"
tX, tY = frame.table("learning", shuff=True, m=5000)

inshape, outshape = frame.neurons_required

# Genome will be the number of hidden neurons
# at each network layer.
ranges = ((2, 60), (2, 60))


def phenotype_to_ann(phenotype):
    net = Network(inshape, layers=[
        Flatten(),
        DenseLayer(int(phenotype[0]), activation="tanh"),
        DenseLayer(int(phenotype[1]), activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ])
    net.finalize(cost="xent", optimizer="adam")
    return net


# Define the fitness function -> evaluate the neural network
def fitness(genotype):
    start = time.time()
    phenotype = to_phenotype(genotype, ranges)
    net = phenotype_to_ann(phenotype)
    net.fit(tX, tY, batch_size=20, epochs=3, verbose=0)
    score = net.evaluate(*frame.table("testing", shuff=True, m=100))[-1]
    timereq = time.time() - start
    return (1. - score), timereq  # fitness is minimized, so we need error rate


pop = Population(
    loci=len(ranges),
    limit=12,
    fitness_function=fitness,
    fitness_weights=[5., 1.])

means, totals, bests = pop.run(epochs=10, verbosity=1,
                               survival_rate=0.7,
                               mutation_rate=0.2,
                               mutation_delta=0.02)
logs = pop.run(epochs=3, verbosity=1,
               survival_rate=0.5,
               mutation_rate=0.0,
               mutation_delta=0.02)

means += logs[0]
totals += logs[1]
bests += logs[2]

print("\nThe winner is:")
winner = phenotype_to_ann(to_phenotype(pop.best, ranges))
winner.describe(1)
winner.fit_csxdata(frame, epochs=60, monitor=["acc"])

Xs = np.arange(1, len(means)+1)
fig, axes = plt.subplots(2, sharex=True)
axes[0].plot(Xs, totals)
axes[0].set_title("Total grade")
axes[1].plot(Xs, means, color="blue")
axes[1].plot(Xs, bests, color="red")
axes[1].set_title("Means and bests")

plt.show()
