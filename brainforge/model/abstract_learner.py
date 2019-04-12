import abc

from brainforge import backend as xp

from .layerstack import LayerStack
from ..costs import cost_functions, CostFunction
from ..util import batch_stream


class Learner:

    def __init__(self, layerstack, cost="mse", name="", **kw):
        if not isinstance(layerstack, LayerStack):
            if "input_shape" not in kw:
                raise RuntimeError("Please supply input_shape as a keyword argument!")
            layerstack = LayerStack(kw["input_shape"], layers=layerstack)
        self.layers = layerstack
        self.name = name
        self.age = 0
        self.cost = cost if isinstance(cost, CostFunction) else cost_functions[cost]

    def fit_generator(self, generator, lessons_per_epoch, epochs=30, classify=True, validation=(), verbose=1, **kw):
        epcosts = []
        lstr = len(str(epochs))
        for epoch in range(1, epochs+1):
            if verbose:
                print("Epoch {:>{w}}/{}".format(epoch, epochs, w=lstr))
            epcosts += self.epoch(generator, no_lessons=lessons_per_epoch, classify=classify,
                                  validation=validation, verbose=verbose, **kw)
        self.age += epochs
        return epcosts

    def fit(self, X, Y, batch_size=20, epochs=30, classify=True, validation=(), verbose=1, shuffle=True, **kw):
        datastream = batch_stream(X, Y, m=batch_size, shuffle=shuffle)
        return self.fit_generator(datastream, len(X), epochs, classify, validation, verbose, **kw)

    def epoch(self, generator, no_lessons, classify=True, validation=None, verbose=1, **kw):

        costs = []
        done = 0
        self.layers.learning = True
        while done < no_lessons:
            batch = next(generator)
            cost = self.learn_batch(*batch, **kw)
            costs.append(cost)

            done += len(batch[0])
            if verbose:
                print("\rDone: {0:>6.1%} Cost: {1: .5f}\t "
                      .format(done/no_lessons, xp.mean(costs)), end="")
        self.layers.learning = False

        if verbose:
            print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(1., xp.mean(costs)), end="")
            if validation:
                self._print_progress(validation, classify)
            print()

        return costs

    def _print_progress(self, validation, classify):
        results = self.evaluate(*validation, classify=classify)

        chain = "Testing cost: {0:.5f}"
        if classify:
            tcost, tacc = results
            accchain = " accuracy: {0:.2%}".format(tacc)
        else:
            tcost = results
            accchain = ""
        print(chain.format(tcost) + accchain, end="")

    def predict(self, X):
        return self.layers.feedforward(X)

    def evaluate(self, X, Y, batch_size=32, classify=True, shuffle=False, verbose=False):
        N = X.shape[0]
        batches = batch_stream(X, Y, m=batch_size, shuffle=shuffle, infinite=False)

        cost, acc = [], []
        for bno, (x, y) in enumerate(batches, start=1):
            if verbose:
                print("\rEvaluating: {:>7.2%}".format((bno*batch_size) / N), end="")
            pred = self.predict(x)
            cost.append(self.cost(pred, y) / len(x))
            if classify:
                pred_classes = xp.argmax(pred, axis=1)
                trgt_classes = xp.argmax(y, axis=1)
                eq = xp.equal(pred_classes, trgt_classes)
                acc.append(eq.mean())
        results = xp.mean(cost)
        if classify:
            results = (results, xp.mean(acc))
        return results

    @abc.abstractmethod
    def learn_batch(self, X, Y, **kw):
        raise NotImplementedError

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def nparams(self):
        return self.layers.nparams
