import numpy as np
from matplotlib import pyplot as plt

from brainforge.evolution import Population


pop = Population(
    loci=5,
    limit=100,
    fitness_function=lambda ind: (np.sum(ind),),
    fitness_weights=[1.],
    mutation_rate=0.1,
    mutation_delta=0.05,
    survivors_rate=0.5
)
log = pop.run(1000, verbosity=0)
print("EVOLUTION: Final grade:   {}".format(pop.grade()))
plt.plot(np.arange(len(log)), log)
plt.show()
