"""
Simple Genetic Algorithm from the perspective of a Biologist
Copyright (C) 2016  Csaba Gór

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
from random import randrange

import numpy as np


# I hereby state that
# this evolutionary algorithm is MINIMIZING the fitness value!


class Population:
    """Model of a population in the ecologigal sense"""

    def __init__(self, loci, fitness_function,
                 fitness_weights,
                 limit=100, survivors_rate=0.5,
                 mutation_rate=0.1, mutation_delta=0.25,
                 force_update_at_every=0):
        """
        :param loci: number of elements in an individual's chromosome
        :param fitness_function: function used to evaluate the "badness" of an individual
        :param limit: maximum number of individuals
        :param survivors_rate: probability of survival
        :param mutation_rate: probability of mutation
        :param mutation_delta: mutation = gene + uniform(+-delta)
        """

        self.fitness = fitness_function
        self.fitness_w = np.array(fitness_weights)
        self.limit = limit
        self.survivors = survivors_rate
        self.mutation_rate = mutation_rate
        self.mutation_delta = mutation_delta

        self.fitnesses = np.zeros((limit, len(fitness_weights)))
        self.grades = np.zeros((limit,))
        self.individuals = np.random.uniform(size=(limit, loci))
        self.force_update = force_update_at_every

        self.age = 0

        self.update()
        print("EVOLUTION: initial grade:", self.grade())

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
            self.grades[ind] = (self.fitnesses[ind] * self.fitness_w).sum()
        if verbose:
            print("\rUpdating {0}/{1}".format(lim, lim))
        self.grades = rescale(self.grades)

    def run(self, epochs, verbosity=1):
        past_grades = []
        for epoch in range(1, epochs+1):
            if verbosity:
                print("-"*50)
                print("Epoch {0:>{w}}/{1}".format(epoch, epochs, w=len(str(epochs))))
            diers = np.random.choice([0., 1.], size=self.grades.shape,
                                     p=[self.survivors, 1.-self.survivors])
            candidates = self.get_candidates()
            candidates = diers[:, None] * candidates + (1.-diers[:, None]) * self.individuals
            mut_mask = np.random.choice([0., 1.],
                                        size=self.individuals.shape,
                                        p=[1.-self.mutation_rate, self.mutation_rate])
            mutations = mut_mask * np.random.uniform(low=-self.mutation_delta,
                                                     high=self.mutation_delta,
                                                     size=self.individuals.shape)
            self.individuals = np.clip(candidates + mutations, 0., 1.)

            inds = np.argwhere(diers + mut_mask.sum(axis=1))
            if self.force_update:
                if epoch % self.force_update != 0:
                    inds = None
            self.update(inds, verbose=verbosity)

            self.age += 1
            if verbosity:
                self.describe(verbosity-1)
            past_grades.append(self.grade())

        print()
        return past_grades

    def get_candidates(self):
        candidates = np.zeros_like(self.individuals)
        rr = lambda: randrange(self.limit)
        i = 0
        while i != self.limit:
            left, right = rr(), rr()
            prob = 1. - (self.grades[left] * self.grades[right])
            if prob > np.random.uniform():
                continue
            new = mate(self.individuals[left], self.individuals[right])
            candidates[i] = new
            i += 1
        return candidates

    def grade(self):
        """Calculates an average fitness value for the whole population"""
        return (self.fitnesses * self.fitness_w[None, :]).sum()

    def describe(self, show=0):
        """Print out useful information about a population"""
        showme = np.argsort(self.grades)[:show]
        chain = "-"*50 + "\n"
        chain += "Population of age {}\n".format(self.age)
        for i, index in enumerate(showme):
            genomechain = ", ".join(
                "{:>6.4f}".format(loc) for loc in
                np.round(self.individuals[index], 4))
            fitnesschain = "[" + ", ".join(
                "{:>8.4f}".format(fns) for fns in
                self.fitnesses[index]) + "]"
            chain += "TOP {:>2}: [{:^14}] F = {:<}\n".format(
                i+1, genomechain, fitnesschain)
        chain += "Size : {}\n".format(self.limit)
        chain += "Avg F: {}".format(self.grade())
        print(chain)

    @property
    def best(self):
        arg = np.argmin(self.grades)
        return self.individuals[arg]


def mate(ind1, ind2):
    """Mate an individual with another"""
    return np.where(np.random.uniform(size=ind1.shape) < 0.5, ind1, ind2)


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
    output *= 0.90
    output += 0.05
    assert np.isclose(output.min(), 0.05) and np.isclose(output.max(), 0.95), \
        "Failed with values: {}, {}".format(output.min(), output.max())
    return output
