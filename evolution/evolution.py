"""
Simple Genetic Algorithm from the perspective of a Biologist
Copyright (C) 2016  Csaba Gór

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import time
import random

from csxdata.utilities.misc import avg, feature_scale, chooseN


# I hereby state that
# this evolutionary algorithm is MINIMIZING the fitness value!


class Population:
    """Model of a population in the ecologigal sense"""

    def __init__(self, limit, survivors_rate, crossing_over_rate, mutation_rate,
                 fitness_function, max_offsprings, ranges, parallel=False, jobs=0):

        self.fitness = fitness_function

        self.limit = limit
        self.survivors = survivors_rate
        self.crossing_over_rate = crossing_over_rate
        self.mutation_rate = mutation_rate
        self.max_offsprings = max_offsprings
        self.ranges = ranges
        self.parallel = parallel

        self.individuals = [Individual(random_genome(ranges)) for _ in range(limit)]

        self.age = 0

        if not parallel:
            self.update(False)
        else:
            self.parallel_update(False, jobs=jobs)

    def run(self, epochs, verbose=1, parallel=False, jobs=None):
        """Runs a given number of epochs: a selection followed by a reproduction"""

        start = time.time()
        for epoch in range(1, epochs + 1):

            if verbose > 1:
                print("-"*50)
                print("Epoch {0:>{w}}/{1}".format(epoch, epochs, w=len(str(epochs))))
            if verbose and epoch % 100 == 0:
                print("-"*50)
                print("Epoch {0:>{w}}/{1}".format(epoch, epochs, w=len(str(epochs))))

            if verbose > 2:
                print("Selection... ", end="")
            self.selection()
            if verbose > 2:
                print("Done!\nReproduction... ", end="")
            self.reproduction()
            if verbose > 2:
                print("Done!\nMutation... ", end="")
            self.mutation()
            if verbose > 2:
                print("Done!")
            if epoch % 5 == 0:
                if not parallel:
                    self.update(quick=False, verbose=verbose)
                else:
                    self.parallel_update(False, verbose, jobs)
            else:
                if not parallel:
                    self.update(quick=True, verbose=verbose)
                else:
                    self.parallel_update(True, verbose, jobs)

            self.age += 1

            while len(self.individuals) < 4:
                if len(self.individuals) < 2:
                    raise RuntimeError("Population died out. Adjust selection parameters!")
                self.reproduction()
                if len(self.individuals) >= 4:
                    print("Added extra reproduction steps due to low number of individuals!")
                    self.update()

            if verbose > 0 and epoch % 100 == 0:
                describe(self, show=1)
            if verbose > 1:
                describe(self, show=1)

        if verbose:
            print("\n-------------------------------")
            print("This took", round(time.time() - start, 2), "seconds!")

        print()

        return self

    def update(self, quick=True, verbose=1):
        N = len(self.individuals)
        for i, ind in enumerate(self.individuals, start=1):
            chain = "\rQuick " if quick else "\rFull "
            if verbose:
                print(chain + "update on fitnesses: {0:>{w}}/{1}"
                      .format(i, N, w=len(str(N))), end="")
            if quick and ind.fitness is None:
                ind.fitness = self.fitness(ind.genome, queue=None)
            elif not quick:
                ind.fitness = self.fitness(ind.genome, queue=None)
        print(" Done!")

    def mapparallel(self, quick, verbose=1, jobs=None):
        import multiprocessing as mp

        print("Doing", end=" ")
        if quick:
            inds = [(ind if ind.fitness is None else None)
                    for ind in self.individuals]
            print("quick", end=" ")
        else:
            inds = self.individuals
            print("full", end=" ")
        print("parallel update!")

        # This hackaround is ugly
        try:
            pool = mp.Pool(processes=jobs)
            fitnesses = pool.map(self.fitness, [ind.genome for ind in inds if ind])
        finally:
            pool.close()
            pool.join()
        assert len(fitnesses) == len(inds) - inds.count(None)
        for i, ind in enumerate(inds):
            if ind:
                self.individuals[i].fitness = fitnesses.pop(0)

        print("Done!")

    def parallel_update(self, quick=True, verbose=1, jobs=0):
        import multiprocessing as mp

        if quick:
            inds = [ind for ind in self.individuals if ind.fitness is None]
        else:
            inds = self.individuals

        size = len(inds)
        queue = mp.Queue()
        new_inds = []
        jobs = jobs if jobs else mp.cpu_count()
        while inds:
            if verbose:
                print("\rParallel updating fitnesses: {0:>{w}}/{1}"
                      .format(len(new_inds), size, w=len(str(size))),
                      end="")

            procs = []
            some_new_inds = []
            workers = jobs if len(inds) >= jobs else len(inds)

            for _ in range(workers):
                ind = inds.pop()
                procs.append(mp.Process(target=self.fitness, args=(ind.genome, queue)))
                procs[-1].start()
            while len(some_new_inds) != workers:
                some_new_inds.append(queue.get())
                time.sleep(0.1)
            for proc in procs:
                proc.join()

            new_inds += some_new_inds

        assert len(new_inds) == size, ("Expected {} individuals, but got {}"
                                       .format(size, len(new_inds)))
        print("\rParallel updating fitnesses: {0:>{w}}/{1}"
              .format(len(new_inds), size, w=len(str(size))))
        self.individuals = list(new_inds)

    def selection(self):
        # ARAAAAAAAAAAAAAAAAAAARRARARAR
        self.individuals = [individual for survival_chance, individual in zip((random.uniform(0.0, fitness) for fitness in feature_scale((ind.fitness for ind in self.individuals), from_=0.05, to=0.95)), self.individuals) if survival_chance < self.survivors]
        # fscaled = feature_scale((individual.fitness for individual in self.individuals), from_=0.05, to=0.95)
        # chances = (random.uniform(0.0, fitness) < self.survivors for fitness in fscaled)
        # self.individuals = [individual for chance, individual in zip(chances, self.individuals) if chance]

    def reproduction(self):
        """This method generates new individuals by mating existing ones"""

        # A reverse-ordered list is generated from the ind fitnesses
        fscaled = feature_scale((ind.fitness for ind in self.individuals))
        reproducers = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)

        # In every round, two individuals with the highest fitnesses reproduce
        while (len(reproducers) > 1) and (len(self.individuals) + self.max_offsprings <= self.limit):
            # Stochastic reproduction
            self.mate(*chooseN(reproducers, n=2))

    def mutation(self):
        """Generate mutations in the given population. Rate is given by <pop.mutation_rate>"""

        mutations = 0

        # All the loci in the population
        size = len(self.individuals)  # The number of individuals
        loci = len(self.individuals[0].genome)  # The number of loci in a single individual
        all_loci = size * loci  # All loci in the population given the number of chromosomes (T) = 1

        # The chance of mutation_rate is applied to loci and not individuals!
        for i in range(all_loci):
            roll = random.random()
            if roll < (self.mutation_rate / loci):
                no_ind = i // loci
                m_loc = i % loci

                # OK, like, WTF???
                newgenome = self.individuals[no_ind].genome[:]  # the genome gets copied
                newgenome[m_loc] = random_locus(self, m_loc)
                self.individuals[no_ind].genome = newgenome
                # Above snippet is a workaround, because the following won't work:
                # self.individuals[i // loci].genome[m_loc] = random_locus(self, m_loc)
                # It somehow alters the selected individual's genome, but fails to reset
                # its fitness to None. Something earie is going on here...
                self.individuals[no_ind].fitness = None
                self.individuals[no_ind].mutant += 1
                mutations += 1

    def mate(self, ind1, ind2):
        """Mate this individual with another"""
        no_offsprings = random.randint(1, self.max_offsprings + 1)
        for _ in range(no_offsprings):
            if random.random() < self.crossing_over_rate:
                newgenomes = crossing_over(ind1.genome, ind2.genome)
                newgenome = random.choice(newgenomes)
            else:
                newgenome = random.choice((ind1.genome, ind2.genome))
            self.individuals.append(Individual(newgenome))

    def grade(self):
        """Calculates an average fitness value for the whole population"""
        return avg([ind.fitness for ind in self.individuals])

    @property
    def best(self):
        return sorted(self.individuals, key=lambda ind: ind.fitness)[-1]


class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None
        self.mutant = 0


def crossing_over(chromosome1, chromosome2):
    # Crossing over works similarily to the biological crossing over
    # A point is selected randomly along the loci of the chromosomes,
    # excluding position 0 and the last position (after the last locus).
    # Whether the "head" or "tail" of the chromosome gets swapped is random
    position = random.randrange(len(chromosome1) - 2) + 1
    if random.random() >= 0.5:
        return (chromosome1[:position] + chromosome2[position:],
                chromosome2[:position] + chromosome1[position:])
    else:
        return (chromosome2[:position] + chromosome1[position:],
                chromosome1[:position] + chromosome2[position:])


def random_genome(ranges):
    return [{float: random.uniform, int: random.randrange, str: wrapchoice}
            [type(t[0])](*t) for t in ranges]


def wrapchoice(*args):
        return random.choice(args)


def random_locus(pop, locus):
    ranges = pop.ranges[locus]
    return {float: random.uniform, int: random.randrange, str: wrapchoice}[type(ranges[0])](*ranges)


def mutants(pop):
    holder = [ind.mutant > 0 for ind in pop.individuals]
    return sum(holder) / len(holder)


def describe(pop, show=0):
    """Print out useful information about a population"""
    showme = sorted(pop.individuals, key=lambda indiv: indiv.fitness)[:show]
    for i, ind in enumerate(showme, start=1):
        print("TOP {}: {} F={}".format(i, ind.genome, ind.fitness))
    print("Size :", len(pop.individuals), sep="\t")
    print("Avg F:", pop.grade(), sep="\t")
