from ..model.layerstack import LayerStack
from ..metrics import costs as _costs, metrics as _metrics
from ..util import batch_stream, logging


class Learner:

    def __init__(self, layerstack, cost="mse", name="", **kw):
        if not isinstance(layerstack, LayerStack):
            if "input_shape" not in kw:
                raise RuntimeError("Please supply input_shape as a keyword argument!")
            layerstack = LayerStack(kw["input_shape"], layers=layerstack)
        self.layers = layerstack
        self.name = name
        self.age = 0
        self.cost = _costs.get(cost)

    def fit_generator(self, generator, lessons_per_epoch, epochs=30, metrics=(), validation=(), verbose=1, **kw):
        metrics = [_metrics.get(metric) for metric in metrics]
        history = logging.MetricLogs.from_metric_list(lessons_per_epoch, ("cost",), metrics)
        lstr = len(str(epochs))
        for epoch in range(1, epochs+1):
            if verbose:
                print("Epoch {:>{w}}/{}".format(epoch, epochs, w=lstr))
            epoch_history = self.epoch(generator, updates_per_epoch=lessons_per_epoch, metrics=metrics,
                                       validation=validation, verbose=verbose, **kw)
            history.update(epoch_history)

        return history

    def fit(self, X, Y, batch_size=20, epochs=30, metrics=(), validation=(), verbose=1, shuffle=True, **kw):
        metrics = [_metrics.get(metric) for metric in metrics]
        datastream = batch_stream(X, Y, m=batch_size, shuffle=shuffle)
        return self.fit_generator(datastream, len(X) // batch_size, epochs, metrics, validation, verbose, **kw)

    def epoch(self, generator, updates_per_epoch, metrics=(), validation=None, verbose=1, **kw):
        metrics = [_metrics.get(metric) for metric in metrics]
        history = logging.MetricLogs.from_metric_list(updates_per_epoch, ["cost"], metrics)
        done = 0

        self.layers.learning = True
        batch_size = 0
        for i in range(updates_per_epoch):
            batch = next(generator)
            batch_size = len(batch[0])
            epoch_metrics = self.learn_batch(*batch, metrics=metrics, **kw)
            history.record(epoch_metrics)
            if verbose:
                history.log(prefix="\r", end="")

        self.layers.learning = False
        if verbose and validation:
            history = self.evaluate(*validation, batch_size=batch_size, metrics=metrics)
            history.log(prefix=" ", suffix="")
        if verbose:
            print()

        self.age += updates_per_epoch
        return history

    def predict(self, X):
        return self.layers.feedforward(X)

    def evaluate_batch(self, x, y, metrics=()):
        m = len(x)
        preds = self.predict(x)
        eval_metrics = {"cost": self.cost(self.output, y) / m}
        if metrics:
            for metric in metrics:
                eval_metrics[str(metric).lower()] = metric(preds, y) / m
        return eval_metrics

    def evaluate(self, X, Y, batch_size=32, metrics=(), verbose=False):
        metrics = [_metrics.get(metric) for metric in metrics]
        N = X.shape[0]
        batch_size = min(batch_size, N)
        steps = int(round(N / batch_size))
        history = logging.MetricLogs.from_metric_list(steps, ["cost"], metrics)

        for x, y in batch_stream(X, Y, m=batch_size, shuffle=False, infinite=False):
            eval_metrics = self.evaluate_batch(x, y, metrics)
            history.record(eval_metrics)
            if verbose:
                history.log("\r", end="")

        if verbose:
            print()
        history.reduce_mean()
        return history

    def learn_batch(self, X, Y, metrics=(), **kw) -> dict:
        raise NotImplementedError

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def outshape(self):
        return self.layers.outshape

    @property
    def num_params(self):
        return self.layers.num_params

    @property
    def trainable_layers(self):
        return self.layers.trainable_layers
