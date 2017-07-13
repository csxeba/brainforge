import numpy as np

from ..util.persistance import Capsule
from ..ops.lowlevel import ops


class Predictor:

    def __init__(self, capsule):
        c = Capsule.read(capsule)

        self.name = "Predictor from " + c["name"]
        self.ops = [ops[arch.split("-")[0]](params) for arch, params
                    in zip(c["architechure"], c["architecture"])]
        self.cost = c["cost"]

    def predict(self, X):
        for op in self.ops:
            X = op(X)
        return X

    def classify(self, X):
        X = self.predict(X)
        return X.argmax(axis=1)

    def evaluate(self, X, Y, costfn, classify=False):
        preds = self.predict(X)
        cost = costfn(preds, Y)

        if classify:
            eq = np.equal(preds.argmax(axis=1), Y.argmax(axis=1))
            return cost, eq.sum() / eq.size()

        return cost
