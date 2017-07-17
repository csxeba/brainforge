import abc

import numpy as np

from .abstract_graph import Graph
from ..util import batch_stream


class Learner(Graph):

    def __init__(self, input_shape, layers=(), name=""):
        super().__init__(input_shape, layers, name)
        self.age = 0
        self.learning = False

    def fit_generator(self, generator, lessons_per_epoch, epochs=30, monitor=(), validation=(), verbose=1):
        self.N = lessons_per_epoch

        epcosts = []
        lstr = len(str(epochs))
        epoch = 1
        while epoch <= epochs:
            if verbose:
                print("Epoch {:>{w}}/{}".format(epoch, epochs, w=lstr))

            epcosts += self.epoch(generator, monitor, validation, verbose)
            epoch += 1

        self.age += epochs
        return epcosts

    def fit(self, X, Y, batch_size=20, epochs=30, monitor=(), validation=(), verbose=1, shuffle=True, w=None):
        if w is not None:
            datastream = batch_stream(X, Y, w, m=batch_size, shuffle=shuffle)
        else:
            datastream = batch_stream(X, Y, m=batch_size, shuffle=shuffle)
        return self.fit_generator(datastream, len(X), epochs, monitor, validation, verbose)

    def epoch(self, generator, monitor, validation, verbose):

        if not self._finalized:
            raise RuntimeError("Architecture not finalized!")

        costs = []
        done = 0.

        self.learning = True
        while round(done, 5) < 1.:
            cost = self.learn_batch(*next(generator))
            cost /= self.m
            costs.append(cost)

            done += self.m / self.N
            if verbose:
                print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(done, np.mean(costs)), end="")
        self.learning = False

        if verbose:
            print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(1., np.mean(costs)), end="")

        if verbose:
            if validation:
                self._print_progress(validation, monitor)
            print()

        return costs

    def _print_progress(self, validation, monitor):
        classificaton = "acc" in monitor
        results = self.evaluate(*validation, classify=classificaton)

        chain = "Testing cost: {0:.5f}"
        if classificaton:
            tcost, tacc = results
            accchain = " accuracy: {0:.2%}".format(tacc)
        else:
            tcost = results
            accchain = ""
        print(chain.format(tcost) + accchain, end="")

    @abc.abstractmethod
    def learn_batch(self, X, Y):
        raise NotImplementedError
