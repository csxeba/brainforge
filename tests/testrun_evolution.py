import numpy as np
from matplotlib import pyplot as plt

from brainforge.evolution import Population


def upscale(ind):
    x = np.array(ind) - 0.5
    x *= 10.
    return x


def fitness(ind):
    return np.prod(ind), np.sum(ind)


def matefn(ind1, ind2):
    return np.where(np.random.uniform())

pop = Population(
    loci=2,
    fitness_function=fitness,
    fitness_weights=[1., 1.],
    mate_function=matefn,
    limit=100)

pop.individuals += 0.5
pop.individuals = np.clip(pop.individuals, 0., 1.)

plt.ion()
obj = plt.plot(*upscale(pop.individuals.T), "ro")[0]
plt.show()
plt.xlim([-5, 5])
plt.ylim([-5, 5])
for i in range(100):
    pop.run(1, verbosity=0)
    obj.set_data(*upscale(pop.individuals.T))
    plt.pause(0.01)
