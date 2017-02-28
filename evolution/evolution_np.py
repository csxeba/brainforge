import warnings
from random import randrange, choice

import numpy as np


# I hereby state that
# this evolutionary algorithm is MINIMIZING the fitness value!


class Population:
    """Model of a population in the ecologigal sense"""

    def __init__(self, loci: int,
                 fitness_function: callable,
                 fitness_weights=None,
                 limit: int=100,
                 grade_function: callable=None,
                 mate_function: callable=None):
        """
        :param loci: number of elements in an individual's chromosome
        :param fitness_function: accepts a genotype, returns a tuple of fitnesses
        :param fitness_weights: used as summation weights if grade_function is not set
        :param limit: maximum number of individuals
        :param grade_function: accepts fitnesses of an individual, returns scalar
        :param mate_function: accepts two genotypes, returns an offspring genotype
        """

        self.fitness = fitness_function
        self.limit = limit

        self.fitnesses = np.zeros((limit, len(fitness_weights)))
        self.grades = np.zeros((limit,))
        if grade_function is None:
            if fitness_weights is None:
                raise RuntimeError("Either supply grade_function or fitness_weights!")
            self.grade_function = self._default_grade_function
        else:
            if fitness_weights is not None:
                warnings.warn("grade_function supplied, fitness_weights ignored!")
            self.grade_function = grade_function

        self.mate_function = (self._default_mate_function
                              if mate_function is None
                              else mate_function)

        self.fitness_w = fitness_weights
        self.individuals = np.random.uniform(size=(limit, loci))

        self.age = 0

        self.update(verbose=1)
        print("EVOLUTION: initial mean grade:", self.mean_grade())
        print("EVOLUTION: initial best grade:", self.grades.min())

    def update(self, inds=None, verbose=0):
        if inds is None:
            inds = np.arange(self.individuals.shape[0])
        else:
            inds = inds.ravel()
        lim = self.limit
        strlen = len(str(lim))
        for ind, gen in zip(inds, self.individuals):
            if verbose:
                print("\rUpdating {0:>{w}}/{1}".format(int(ind)+1, lim, w=strlen), end="")
            self.fitnesses[ind] = self.fitness(gen)
            self.grades[ind] = self.grade_function(self.fitnesses[ind])
        if verbose:
            print("\rUpdating {0}/{1}".format(lim, lim))

    def get_candidates(self, survivors=None):
        prbs = rescale(self.grades)
        candidates = np.zeros_like(self.individuals)
        if survivors is None:
            rr = lambda: (randrange(self.limit), randrange(self.limit))
        else:
            rr = lambda: (choice(survivors), choice(survivors))
        i = 0
        while i != self.limit:
            left, right = rr()
            prob = np.mean([prbs[left], prbs[right]])
            if prob > np.random.uniform():
                continue
            new = self.mate_function(left, right)
            candidates[i] = new
            i += 1
        return candidates

    def selection(self, rate):
        survmask = np.zeros_like(self.grades)
        if rate:
            survivors = np.argsort(self.grades)[:int(self.limit * rate)]
            survmask[survivors] = 1.
        return survmask

    def mutation(self, rate):
        if rate:
            mut_mask = np.random.choice([0., 1.],
                                        size=self.individuals.shape,
                                        p=[1. - rate, rate])
            mutations = mut_mask * np.random.uniform(low=0., high=1.,
                                                     size=self.individuals.shape)
        else:
            mutations = np.zeros_like(self.individuals)
        return mutations

    def run(self, epochs: int,
            survival_rate: float=0.5,
            mutation_rate: float=0.1,
            force_update_at_every: int=0,
            verbosity: int=1):
        """
        Runs the algorithm, optimizing the individuals.

        :param epochs: number of epochs to run for
        :param survival_rate: 0-1, how many individuals survive the selection
        :param mutation_rate: 0-1, rate of mutation at each epoch
        :param force_update_at_every: complete reupdate at specified intervals
        :param verbosity: 1 is verbose, < 1 also prints out v - 1 individuals
        :return: means, stds, bests (grades at each epoch)
        """

        mean_grades = []
        grades_std = []
        bests = []
        epln = len(str(epochs))
        for epoch in range(1, epochs+1):
            if verbosity:
                print("-"*50)
                print("Epoch {0:>{w}}/{1}".format(epoch, epochs, w=epln))

            survmask = self.selection(survival_rate)
            candidates = self.get_candidates(survivors=np.argwhere(survmask))
            newgen = ((1. - survmask)[:, None] * candidates +
                      survmask[:, None] * self.individuals)
            mutations = self.mutation(mutation_rate)

            self.individuals = newgen + mutations

            if force_update_at_every and epoch % force_update_at_every == 0:
                inds = None
            else:
                inds = np.argwhere(survmask + mutations.sum(axis=1))

            self.update(inds, verbose=verbosity)

            if verbosity:
                self.describe(verbosity-1)
            mean_grades.append(self.grades.mean())
            grades_std.append(self.grades.std())
            bests.append(self.grades.min())

            self.age += 1

        print()
        return np.array(mean_grades), np.array(grades_std), np.array(bests)

    def total_grade(self):
        return self.grades.sum()

    def mean_grade(self):
        """Calculates an average fitness value for the whole population"""
        return self.grades.std()

    def describe(self, show=0):
        showme = np.argsort(self.grades)[:show]
        chain = "-"*50 + "\n"
        shln = len(str(show))
        for i, index in enumerate(showme, start=1):
            genomechain = ", ".join(
                "{:>6.4f}".format(loc) for loc in
                np.round(self.individuals[index], 4))
            fitnesschain = "[" + ", ".join(
                "{:^8.4f}".format(fns) for fns in
                self.fitnesses[index]) + "]"
            chain += "TOP {:>{w}}: [{:^14}] F = {:<} G = {:.4f}\n".format(
                i, genomechain, fitnesschain, self.grades[index],
                w=shln)
        bestman = self.grades.argmin()
        chain += "Best Grade : {:7>.4f} ".format(self.grades[bestman])
        chain += "Fitnesses: ["
        chain += ", ".join("{}".format(f) for f in self.fitnesses[bestman])
        chain += "]\n"
        chain += "Mean Grade : {:7>.4f}, STD: {:7>.4f}\n"\
                 .format(self.grades.mean(), self.grades.std())
        print(chain)

    @property
    def best(self):
        arg = np.argmin(self.grades)
        return self.individuals[arg]

    @staticmethod
    def _default_mate_function(gen1, gen2):
        return np.where(np.random.uniform(size=gen1.shape) < 0.5, gen1, gen2)

    def _default_grade_function(self, fitnesses):
        return np.dot(fitnesses, self.fitness_w)


def to_phenotype(ind, ranges):
    if len(ranges) != ind.shape[0]:
        raise RuntimeError("Specified ranges are incompatible with the supplied individuals")
    phenotype = ind.copy()
    for locus, (mini, maxi) in enumerate(ranges):
        phenotype[locus] *= (maxi - mini)
        phenotype[locus] += mini
    return phenotype


def rescale(vector):
    output = vector - vector.min()
    output /= output.max()
    output *= 0.95
    output += 0.05
    return output
