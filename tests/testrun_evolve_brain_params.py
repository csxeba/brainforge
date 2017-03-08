import numpy as np

from matplotlib import pyplot as plt

from brainforge import Network
from brainforge.layers import DenseLayer
from brainforge.evolution import Population

from csxdata import CData
from csxdata.utilities.vectorops import upscale


# 2 fitness values are being used, the classification error rate
# and the L2 norm of the weights. The weight assigned to L2 is
# roughly equivalent to the lambda term in L2 weight regularization.
fweights = (1., 0.1)


def fitness(geno):
    pheno = upscale(geno, -10., 10.)
    net.set_weights(pheno)
    cost, acc = net.evaluate(*frame.table("learning"))
    class_error = 1. - acc
    l2term = np.linalg.norm(pheno) / (35000. * 2)
    fness = class_error, l2term
    return fness

frame = CData(source="/data/Prog/data/csvs/grapes.csv", indeps=6, headers=1, feature="szin",
              lower=True, dehun=True, decimal=True)
frame.transformation = "std"

# We only need a single net, because the Network.set_weights(fold=True)
# can be used to set the network weights to the evolved parameters.
net = Network(frame.neurons_required[0], layers=[
        DenseLayer(30, activation="tanh"),
        DenseLayer(frame.neurons_required[1], activation="sigmoid")
    ])
net.finalize(cost="mse", optimizer="adam")

pop = Population(
    limit=300,
    loci=net.nparams,
    fitness_function=fitness,
    fitness_weights=fweights,
)

means, stds, bests = pop.run(epochs=300,
                             survival_rate=0.0,
                             mutation_rate=0.01,
                             verbosity=0)
# Plot the run dynamics
Xs = np.arange(1, len(means)+1)
plt.plot(Xs, means, color="blue")
plt.plot(Xs, means+stds, color="green", linestyle="--")
plt.plot(Xs, means-stds, color="green", linestyle="--")
plt.plot(Xs, bests, color="red")
plt.title("Mean (blue) and best (red) grades")  # Yes, I'm lazy
plt.show()
