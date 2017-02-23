import numpy as np

from matplotlib import pyplot as plt

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.evolution import Population

from csxdata import CData, roots
from csxdata.utilities.vectorops import upscale

BRAINS = 12
NIND = 300
fweights = [1, 1]


def fitness(geno):
    pheno = upscale(geno, -10., 10.)
    net.set_weights(pheno)
    cost, acc = net.evaluate(*frame.table("learning", m=35000))
    fness = 1. - acc, geno.sum()
    return fness


def build_net(inshp, outshp):
    model = Network(inshp, layers=[
        DenseLayer(30, activation="tanh"),
        DenseLayer(outshp, activation="sigmoid")
    ])
    model.finalize(cost="mse", optimizer="adam")
    return model

frame = CData(roots["misc"] + "mnist.pkl.gz", cross_val=0.0,
              indeps=0, headers=0, fold=False)

net = build_net(*frame.neurons_required)
pop = Population(
    limit=30,
    loci=net.get_weights().size,
    fitness_function=fitness,
    fitness_weights=fweights
)
means, totals, bests = pop.run(epochs=100,
                               survival_rate=0.8,
                               mutation_rate=0.01)
Xs = np.arange(1, len(means)+1)
fig, axarr = plt.subplots(2, sharey=True)
axarr[0].plot(Xs, totals)
axarr[1].plot(Xs, means, color="blue")
axarr[1].plot(Xs, bests, color="red")
axarr[0].set_title("Total grade")
axarr[1].set_title("Mean (blue) and best (red) grades")
plt.tight_layout()
plt.show()
