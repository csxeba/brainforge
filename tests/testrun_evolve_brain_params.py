import numpy as np

from matplotlib import pyplot as plt

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.evolution import Population

from csxdata import CData, roots
from csxdata.utilities.vectorops import upscale

NIND = 300
fweights = [1, 1]


def fitness(geno):
    pheno = upscale(geno, -10., 10.)
    net.set_weights(pheno)
    cost, acc = net.evaluate(*frame.table("learning", m=100))
    fness = 1. - acc, np.linalg.norm(pheno)
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
    limit=300,
    loci=net.get_weights().size,
    fitness_function=fitness,
    fitness_weights=fweights,
    grade_function=lambda *ph: np.prod(np.array(ph))
)
means, totals, bests = pop.run(epochs=200,
                               survival_rate=0.5,
                               mutation_rate=0.0)
Xs = np.arange(1, len(means)+1)
fig, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(Xs, totals)
axarr[1].plot(Xs, means, color="blue")
axarr[1].plot(Xs, bests, color="red")
axarr[0].set_title("Total grade")
axarr[1].set_title("Mean (blue) and best (red) grades")
plt.tight_layout()
plt.show()
