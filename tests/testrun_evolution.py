import time

from evolution.evolution_np import Population


def fn(genome):
    from csxdata.utilities.misc import euclidean

    target = [50] * len(genome)
    return euclidean(genome, target)


def test_evolution():
    limit = 1000
    survivors = 0.4
    crossing_over_rate = 0.2
    mutation_rate = 0.01
    mutation_delta = 0.1
    max_offsprings = 3
    epochs = 300
    fitness = fn
    genome_len = 10

    ranges = [(0, 100) for _ in range(genome_len)]

    demo_pop = Population(limit=limit,
                          survivors_rate=survivors,
                          crossing_over_rate=crossing_over_rate,
                          mutation_rate=mutation_rate,
                          mutation_delta=mutation_delta,
                          fitness_function=fitness,
                          max_offsprings=max_offsprings,
                          ranges=ranges)

    print("Population created with {} individuals".format(len(demo_pop.individuals)))
    demo_pop.describe(3)
    demo_pop.run(epochs)
    print("Run done.")


if __name__ == '__main__':
    start = time.time()
    test_evolution()
    print("Time elapsed: {} s".format(round(time.time()-start, 2)))
