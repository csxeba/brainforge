from random import randrange

import numpy as np


# I hereby state that
# this evolutionary algorithm is MINIMIZING the fitness value!


class Population:
    """Model of a population in the ecologigal sense"""

    def __init__(self, loci,
                 fitness_function,
                 fitness_weights,
                 limit=100,
                 grade_function=None,
                 mate_function=None):
        """
        :param loci: number of elements in an individual's chromosome
        :param fitness_function: function used to evaluate the "badness" of an individual
        :param limit: maximum number of individuals
        """

        self.fitness = fitness_function
        self.fitness_w = np.array(fitness_weights)
        self.limit = limit

        self.fitnesses = np.zeros((limit, len(fitness_weights)))
        self.grades = np.zeros((limit,))
        self.grade_function = (self._default_grade_function
                               if grade_function is None
                               else grade_function)
        self.mate_function = (self._default_mate_function
                              if mate_function is None
                              else mate_function)
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
        for ind, gen in zip(inds, self.individuals):
            if verbose:
                print("\rUpdating {0:>{w}}/{1}".format(int(ind)+1, lim, w=len(str(lim))), end="")
            self.fitnesses[ind] = self.fitness(gen)
            self.grades[ind] = self.grade_function(ind)
        if verbose:
            print("\rUpdating {0}/{1}".format(lim, lim))

    def run(self, epochs, verbosity=1,
            survival_rate=0.5, mutation_rate=0.1,
            force_update_at_every=0):

        mean_grades = []
        total_grades = []
        bests = []
        for epoch in range(1, epochs+1):
            if verbosity:
                print("-"*50)
                print("Epoch {0:>{w}}/{1}".format(epoch, epochs, w=len(str(epochs))))
            prbs = rescale(self.grades)
            survive_pb = np.random.uniform(high=prbs, size=self.grades.shape)
            survivors = np.less_equal(survive_pb, survival_rate)
            candidates = self.get_candidates(prbs)
            candidates = ((1. - survivors)[:, None] * candidates +
                          survivors[:, None] * self.individuals)
            mut_mask = np.random.choice([0., 1.],
                                        size=self.individuals.shape,
                                        p=[1.-mutation_rate, mutation_rate])
            mutations = mut_mask * np.random.uniform(low=0., high=1., size=self.individuals.shape)

            self.individuals = candidates + mutations

            inds = np.argwhere(survivors + mut_mask.sum(axis=1))
            if force_update_at_every:
                if epoch % force_update_at_every == 0:
                    inds = None
            self.update(inds, verbose=verbosity)

            self.age += 1
            if verbosity:
                self.describe(verbosity-1)
            mean_grades.append(self.mean_grade())
            total_grades.append(self.total_grade())
            bests.append(self.grades.min())

        print()
        return mean_grades, total_grades, bests

    def get_candidates(self, prbs):
        candidates = np.zeros_like(self.individuals)
        rr = lambda: (randrange(self.limit), randrange(self.limit))
        i = 0
        while i != self.limit:
            left, right = rr()
            prob = 1. - (prbs[left] * prbs[right])
            if prob > np.random.uniform():
                continue
            new = self.mate_function(left, right)
            candidates[i] = new
            i += 1
        return candidates

    def total_grade(self):
        return self.grades.sum()

    def mean_grade(self):
        """Calculates an average fitness value for the whole population"""
        return self.grades.std()

    def describe(self, show=0):
        """Print out useful information about a population"""
        showme = np.argsort(self.grades)[:show]
        chain = "-"*50 + "\n"
        for i, index in enumerate(showme):
            genomechain = ", ".join(
                "{:>6.4f}".format(loc) for loc in
                np.round(self.individuals[index], 4))
            fitnesschain = "[" + ", ".join(
                "{:^8.4f}".format(fns) for fns in
                self.fitnesses[index]) + "]"
            chain += "TOP {:>2}: [{:^14}] F = {:<}\n".format(
                i+1, genomechain, fitnesschain)
        chain += "Best Grade : {:7>.4f} ".format(self.grades.min())
        chain += "Fitnesses: ["
        chain += ", ".join("{}".format(f) for f in self.fitnesses[self.grades.argmin()])
        chain += "]\n"
        chain += "Mean Grade : {:7>.4f}, STD: {:7>.4f}\n"\
                 .format(self.grades.mean(), self.grades.std())
        print(chain)

    @property
    def best(self):
        arg = np.argmin(self.grades)
        return self.individuals[arg]

    def _default_mate_function(self, ind1, ind2):
        gen1, gen2 = self.individuals[ind1], self.individuals[ind2]
        return np.where(np.random.uniform(size=gen1.shape) < 0.5, gen1, gen2)

    def _default_grade_function(self, ind):
        return (self.fitnesses[ind] * self.fitness_w).sum()


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
    assert np.isclose(output.min(), 0.05) and np.isclose(output.max(), 1.), \
        "Failed with values: {}, {}".format(output.min(), output.max())
    return output
