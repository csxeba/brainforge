import numpy as np
from matplotlib import pyplot as plt

from brainforge.evolution import Population


def fitness(ind):
    return ind.sum()


pop = Population(
    loci=5,
    limit=30,
    fitness_function=fitness,
    fitness_weights=[1.],
    mutation_rate=0.01,
    mutation_delta=0.05,
    selection_pressure=0.5
)
log, bests = pop.run(1000, verbosity=0)
print("EVOLUTION: Final grade:   {}".format(pop.grade()))
print("EVOLUTION: Final best :   {}".format(fitness(pop.best)))
plt.plot(np.arange(len(log)), log)
plt.plot(np.arange(len(log)), np.array(bests).sum(axis=1))
plt.show()
