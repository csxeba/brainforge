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

import numpy as np

from csxdata.utilities.vectorops import featscale, upscale


# I hereby state that
# this evolutionary algorithm is MINIMIZING the fitness value!


class Population:
    """Model of a population in the ecologigal sense"""

    def __init__(self, limit, survivors_rate, crossing_over_rate, mutation_rate,
                 mutation_delta, fitness_function, max_offsprings, ranges):

        self.fitness = fitness_function
        self._fitness_func = fitness_function
        self.limit = limit
        self.survivors = survivors_rate
        self.crossing_over_rate = crossing_over_rate
        self.mutation_rate = mutation_rate
        self.mutation_delta = mutation_delta
        self.max_offsprings = max_offsprings
        self.ranges = ranges

        self.fitnesses = np.zeros((limit,))
        self.individuals = np.random.uniform(size=(limit, len(ranges)))

        self.age = 0

        self.update()
        print("EVOLUTION: initial grade:", self.grade())

    def update(self, inds=None):
        if inds is None:
            inds = np.arange(self.individuals.shape[0])
        else:
            inds = inds.ravel()
        lim = self.limit
        phenotype = np.copy(self.individuals[inds])
        for locus in range(len(self.ranges)):
            mini, maxi = self.ranges[locus]
            phenotype[:, locus] = upscale(phenotype[:, locus], mini, maxi)
        for ind, phen in zip(inds, phenotype):
            print("\rUpdating {0:>{w}}/{1}".format(int(ind)+1, lim, w=len(str(lim))), end="")
            self.fitnesses[ind] = self.fitness(phen)
        print("\rUpdating {0}/{1}".format(lim, lim))

    def run(self, epochs):
        for epoch in range(1, epochs+1):
            print("-"*50)
            print("Epoch {0:>{w}}/{1}".format(epoch, epochs, w=len(str(epochs))))
            diers = np.random.choice([0., 1.], size=self.fitnesses.shape,
                                     p=[self.survivors, 1.-self.survivors])
            candidates = self.get_candidates()
            candidates = diers[:, None] * candidates + (1.-diers[:, None]) * self.individuals
            mut_mask = np.random.choice([0., 1.], size=self.individuals.shape,
                                        p=[1.-self.mutation_rate, self.mutation_rate])
            mutations = mut_mask * np.random.uniform(low=-self.mutation_delta, high=self.mutation_delta,
                                                     size=self.individuals.shape)
            self.individuals = np.clip(candidates + mutations, 0., 1.)
            self.update((np.argwhere(diers + mut_mask.sum(axis=1))
                         if epoch % 5 != 0 else None))
            self.age += 1
            self.describe(3)

    def get_candidates(self):
        candidates = np.zeros_like(self.individuals)
        args = self.fitnesses.argsort()
        pairs = ((left, right, self.fitnesses[left]*self.fitnesses[right])
                 for i, left in enumerate(args) for j, right in enumerate(args)
                 if i != j)
        i = 0
        for left, right, prob in pairs:
            if prob > np.random.uniform():
                continue
            if i == self.limit:
                break
            new = mate(self.individuals[left], self.individuals[right])
            candidates[i] = new
            i += 1
        return candidates

    def grade(self):
        """Calculates an average fitness value for the whole population"""
        return self.fitnesses.mean()

    def describe(self, show=0):
        """Print out useful information about a population"""
        showme = np.argsort(self.fitnesses)[:show]
        chain = "-"*50 + "\n"
        chain += "Population of age {}\n".format(self.age)
        for i, index in enumerate(showme):
            genomechain = ", ".join(str(loc) for loc in np.round(self.individuals[index], 4))
            chain += "TOP {:>2}: [{:^14}] F = {:<}\n".format(
                i+1, genomechain, round(self.fitnesses[index], 4))
        chain += "Size : {}\n".format(self.limit)
        chain += "Avg F: {}".format(self.grade())
        print(chain)


def mate(ind1, ind2):
    """Mate an individual with another"""
    return np.where(np.random.uniform(size=ind1.shape) < 0.5, ind1, ind2)
