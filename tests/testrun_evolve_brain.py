import time

from brainforge import Network
from brainforge.layers import DenseLayer, Flatten
from brainforge.evolution import Population, to_phenotype

from csxdata import CData, roots

dataroot = roots["misc"] + "mnist.pkl.gz"
frame = CData(dataroot, headers=None)
frame.transformation = "std"

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
    net.fit_csxdata(frame, batch_size=20, epochs=3, verbose=0)
    score = net.evaluate(*frame.table("testing", m=10))[-1]
    timereq = time.time() - start
    return (1. - score), timereq  # fitness is minimized, so we need error rate


pop = Population(
    loci=len(ranges),
    limit=30,
    fitness_function=fitness,
    fitness_weights=[1]
).run(epochs=30, verbosity=0)

best = to_phenotype(pop.best, ranges)
