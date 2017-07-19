import warnings

import numpy as np

from ..util import floatX, zX, zX_like


# I solemly state that  # Thanks Mark
# this evolutionary algorithm is MINIMIZING the fitness value!


class Population:

    """
    Model of a population in the ecologigal sense
    """

    def __init__(self, loci: int,
                 fitness_function: callable,
                 fitness_weights=None,
                 limit: int=100,
                 grade_function: callable=None,
                 mate_function: callable=None,
                 mutate_function: callable=None):
        """
        :param loci: number of elements in an individual's chromosome
        :param fitness_function: accepts a genotype, returns a tuple of fitnesses
        :param fitness_weights: used as summation weights if grade_function is not set
        :param limit: maximum number of individuals
        :param grade_function: accepts fitnesses of an individual, returns scalar
        :param mate_function: accepts two genotypes, returns an offspring genotype
        :param mutate_function: accepts individuals and mutation rate, returns mutants
         and index of mutants
        """

        self.fitness = fitness_function
        self.limit = limit

        self.grades = zX(limit)
        if grade_function is None:
            if fitness_weights is None:
                fitness_weights = np.array([1.])
            self.grade_function = self._default_grade_function
        else:
            if fitness_weights is not None:
                warnings.warn("grade_function supplied, fitness_weights ignored!")
            self.grade_function = grade_function

        self.mate_function = (self._default_mate_function
                              if mate_function is None
                              else mate_function)
        self.mutation = (self._default_mutate_function
                         if mutate_function is None
                         else mutate_function)

        self.fitnesses = zX(limit, len(fitness_weights))
        self.fitness_w = fitness_weights
        self.individuals = np.random.uniform(size=(limit, loci)).astype(floatX)

        self.age = 0

        self.initialized = False

    @classmethod
    def from_capsule(cls, capsule, fitness_fn,
                     grade_fn=None, mate_fn=None,
                     mutate_fn=None):
        indiv = capsule["individuals"]
        grades = capsule["grades"]
        fitness_w = capsule["fitness_w"]
        age = capsule["age"]

        limit, loci = indiv.shape
        pop = cls(loci, fitness_fn, fitness_w, limit,
                  grade_fn, mate_fn, mutate_fn)
        pop.individuals = indiv
        pop.grades = grades
        pop.age = age
        return pop

    def update(self, inds=None, verbose=0, *fitness_args, **fitness_kw):
        if inds is None:
            inds = np.arange(self.individuals.shape[0])
        else:
            inds = inds.ravel()
        lim = self.limit
        strlen = len(str(lim))
        for ind in inds:
            gen = self.individuals[ind]
            if verbose:
                print("\rUpdating {0:>{w}}/{1}".format(int(ind)+1, lim, w=strlen), end="")
            self.fitnesses[ind] = self.fitness(gen, *fitness_args, **fitness_kw)
            self.grades[ind] = self.grade_function(self.fitnesses[ind])
        if verbose:
            print("\rUpdating {0}/{1}".format(lim, lim))

    def get_candidates(self, survivors=None):

        if isinstance(survivors, tuple):
            survivors = survivors[0]

        def indstream():
            counter = 0
            lmt = len(survivors)
            arg1, arg2 = np.copy(survivors), np.copy(survivors)
            np.random.shuffle(arg1)
            np.random.shuffle(arg2)
            while 1:
                if counter >= lmt:
                    counter = 0
                    arg1, arg2 = np.copy(survivors), np.copy(survivors)
                    np.random.shuffle(arg1)
                    np.random.shuffle(arg2)
                yield arg1[counter], arg2[counter]
                counter += 1

        prbs = rescale(self.grades)
        candidates = zX_like(self.individuals)
        if survivors is None or survivors.size == 0:
            survivors = np.where(self.selection(0.3))[0]
        i = 0
        for left, right in indstream():
            if i >= self.limit:
                break
            prob = np.mean([prbs[left], prbs[right]])
            roll = np.random.uniform()
            if prob < roll:
                continue
            new = self.mate_function(self.individuals[left], self.individuals[right])
            candidates[i] = new
            i += 1
        return candidates

    def selection(self, rate):
        survmask = zX_like(self.grades)
        if rate:
            survivors = np.argsort(self.grades)[:int(self.limit * rate)]
            survmask[survivors] = 1.
        return survmask

    def run(self, epochs: int,
            survival_rate: float=0.5,
            mutation_rate: float=0.1,
            force_update_at_every: int=0,
            verbosity: int=1,
            *fitness_args, **fitness_kwargs):
        """
        Runs the algorithm, optimizing the individuals.

        :param epochs: number of epochs to run for
        :param survival_rate: 0-1, how many individuals survive the selection
        :param mutation_rate: 0-1, rate of mutation at each epoch
        :param force_update_at_every: complete reupdate at specified intervals
        :param verbosity: 1 is verbose, < 1 also prints out v - 1 individuals
        :param fitness_args: additional arguments for the fitness function
        :param fitness_kwargs: keyword arguments for the fitness function
        :return: means, stds, bests (grades at each epoch)
        """

        if not self.initialized:
            if verbosity:
                print("EVOLUTION: initial update...")
            self.update(verbose=verbosity, *fitness_args, **fitness_kwargs)
            if verbosity:
                print("EVOLUTION: initial mean grade :", self.grades.mean())
                print("EVOLUTION: initial std of mean:", self.grades.std())
                print("EVOLUTION: initial best grade :", self.grades.min())

        mean_grades = []
        grades_std = []
        bests = []
        epln = len(str(epochs))
        mutation_rate /= self.individuals.shape[-1]
        for epoch in range(1, epochs+1):
            if verbosity:
                print("-"*50)
                print("Epoch {0:>{w}}/{1}".format(epoch, epochs, w=epln))

            survmask = self.selection(survival_rate)
            candidates = self.get_candidates(survivors=np.where(survmask))
            newgen = np.where(survmask[:, None], self.individuals, candidates)
            newgen, mutants = self.mutation(mutation_rate, newgen)

            self.individuals = newgen

            if force_update_at_every and epoch % force_update_at_every == 0:
                inds = None
            else:
                inds = np.where(survmask)[0]
                inds = np.append(inds, mutants)
                inds = np.unique(inds)

            self.update(inds, verbose=verbosity, *fitness_args, **fitness_kwargs)

            if verbosity:
                self.describe(verbosity-1)

            mean_grades.append(self.grades.mean())
            grades_std.append(self.grades.std())
            bests.append(self.grades.min())

            self.age += 1

        if verbosity:
            print()
        return np.array(mean_grades), np.array(grades_std), np.array(bests)

    def total_grade(self):
        return self.grades.sum()

    def mean_grade(self):
        """Calculates an average fitness value for the whole population"""
        return self.grades.mean()

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

    @staticmethod
    def _default_mutate_function(rate, individuals):
        if rate:
            mut_mask = np.random.uniform(size=individuals.shape) < rate
            mutants = np.where(mut_mask.sum(axis=1))[0]
            mutations = np.random.uniform(size=(mut_mask.sum(),))
            individuals[np.nonzero(mut_mask)] = mutations
        else:
            mutants = np.array([], dtype=int)
        return individuals, mutants

    def capsule(self):
        return dict(age=self.age, individuals=self.individuals,
                    grades=self.grades, fitness_w=self.fitness_w)


def to_phenotype(ind, ranges):
    if len(ranges) != ind.shape[0]:
        raise RuntimeError("Specified ranges are incompatible with the supplied individuals")
    phenotype = ind.copy()
    for locus, (mini, maxi) in enumerate(ranges):
        phenotype[locus] *= (maxi - mini)
        phenotype[locus] += mini
    return phenotype


def rescale(vector):
    output = (vector - vector.min())
    output /= output.max() + 1e-8
    output *= 0.95
    output += 0.05
    return output
