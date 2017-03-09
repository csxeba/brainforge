import numpy as np
from matplotlib import pyplot as plt

from brainforge.evolution import Population


def upscale(ind):
    x = ind * 10.
    return x


def fitness(ind):
    return np.sqrt(np.sum(np.square(ind))),


def matefn1(ind1, ind2):
    return np.where(np.random.uniform() < 0.5, ind1, ind2)


def matefn2(ind1, ind2):
    return np.add(ind1, ind2) / 2.


pop = Population(
    loci=2,
    fitness_function=fitness,
    fitness_weights=[1.],
    mate_function=matefn2,
    limit=100)

plt.ion()
obj = plt.plot(*upscale(pop.individuals.T), "ro", markersize=2)[0]
plt.xlim([-1, 11])
plt.ylim([-1, 11])

X, Y = np.linspace(-1, 11, 10), np.linspace(-1, 11, 10)
X, Y = np.meshgrid(X, Y)
Z = np.array([fitness([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()
for i in range(100):
    pop.run(1, verbosity=1, survival_rate=0.9)
    obj.set_data(*upscale(pop.individuals.T))
    plt.pause(0.25)
