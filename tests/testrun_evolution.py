import numpy as np
from matplotlib import pyplot as plt

from brainforge.evolution import Population


def fitness(ind):
    return np.array([ind.sum()])

pop = Population(
    loci=5,
    fitness_function=fitness,
    fitness_weights=[1.],
    limit=1000)

means, stds, bests = pop.run(100, verbosity=0,
                             survival_rate=0.33,
                             mutation_rate=0.0)
print("EVOLUTION: Final grade:   {}".format(pop.mean_grade()))
print("EVOLUTION: Final best :   {}".format(fitness(pop.best)[0]))
Xs = np.arange(1, len(means)+1)

pop.describe(3)

plt.title("Run dynamics: means (blue), std (red), bests (green)")
plt.plot(Xs, means, color="blue")
plt.plot(Xs, means+stds, color="red")
plt.plot(Xs, means-stds, color="red")
plt.plot(Xs, bests, color="green")
plt.show()
