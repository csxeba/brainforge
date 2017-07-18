import numpy as np


def mutate(individuals, rate):
    mut_mask = np.random.uniform(size=individuals.shape) < rate
    mutants = np.where(mut_mask.sum(axis=1))[0]
    mutations = np.random.uniform(size=(mut_mask.sum(),))
    individuals[np.nonzero(mut_mask)] = mutations
