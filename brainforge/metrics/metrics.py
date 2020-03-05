import numpy as np


class Metric:

    def __call__(self, outputs, targets):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__.lower().replace("_", "")


class _Accuracy(Metric):

    def __call__(self, outputs, targets):
        return np.sum(outputs.argmax(axis=1) == targets.argmax(axis=1))


accuracy = _Accuracy()
acc = accuracy

_metrics = {k: v for k, v in locals().items() if k[0] != "_" and k != "Metric"}


def get(metric):
    if isinstance(metric, Metric):
        return metric
    metric = _metrics.get(metric)
    if metric is None:
        raise ValueError("No such metric: {}".format(metric))
    return metric
