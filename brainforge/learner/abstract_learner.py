import time

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

    def fit_generator(self,
                      generator,
                      lessons_per_epoch,
                      epochs=30,
                      metrics=(),
                      validation=(),
                      validation_steps=None,
                      verbose=1, **kw):

        metrics = [_metrics.get(metric) for metric in metrics]
        history = logging.MetricLogs.from_metric_list(lessons_per_epoch, ("cost",), metrics)
        lstr = len(str(epochs))
        for epoch in range(1, epochs+1):
            if verbose:
                print("Epoch {:>{w}}/{}".format(epoch, epochs, w=lstr))
            epoch_history = self.epoch(generator, updates_per_epoch=lessons_per_epoch, metrics=metrics,
                                       validation=validation, validation_steps=validation_steps, verbose=verbose, **kw)
            history.update(epoch_history)

        return history

    def fit(self, X, Y,
            batch_size=20,
            epochs=30,
            metrics=(),
            validation=(),
            validation_steps=None,
            verbose=1,
            shuffle=True,
            **kw):

        metrics = [_metrics.get(metric) for metric in metrics]
        datastream = batch_stream(X, Y, m=batch_size, shuffle=shuffle)
        return self.fit_generator(datastream, len(X) // batch_size, epochs, metrics, validation, validation_steps,
                                  verbose, **kw)

    def epoch(self, generator, updates_per_epoch, metrics=(), validation=None, validation_steps=None, verbose=1, **kw):
        start = time.time()

        metrics = [_metrics.get(metric) for metric in metrics]
        history = logging.MetricLogs.from_metric_list(updates_per_epoch, ["cost"], metrics)

        self.layers.learning = True
        batch_size = 0
        for i in range(updates_per_epoch):
            batch = next(generator)
            batch_size = len(batch[0])
            epoch_metrics = self.learn_batch(*batch, metrics=metrics, **kw)
            history.record(epoch_metrics)
            if verbose:
                history.log(prefix="\rTraining ", end="", add_progress=True)

        self.layers.learning = False
        if verbose and validation:
            if type(validation) in (tuple, list):
                eval_history = self.evaluate(*validation, batch_size=batch_size, metrics=metrics, verbose=False)
            else:
                if validation_steps is None:
                    raise RuntimeError("If validating on a stream, validation_steps must be set to a positive integer.")
                eval_history = self.evaluate_stream(validation, validation_steps, metrics, verbose=False)
            eval_history.log(prefix=" Validation ", suffix="", add_progress=False)

        if verbose:
            print(f" took {time.time() - start // 60} minutes")

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

    def evaluate_stream(self, stream, steps, metrics=(), verbose=False):
        history = logging.MetricLogs.from_metric_list(steps, ["cost"], metrics)
        metrics = [_metrics.get(metric) for metric in metrics]
        for i, (x, y) in enumerate(stream, start=1):
            eval_metrics = self.evaluate_batch(x, y, metrics)
            history.record(eval_metrics)
            if verbose:
                history.log("\r", end="")
            if i >= steps:
                break
        if verbose:
            print()
        history.reduce_mean()
        return history

    def evaluate(self, X, Y, batch_size=32, metrics=(), verbose=False):
        N = X.shape[0]
        batch_size = min(batch_size, N)
        steps = int(round(N / batch_size))

        stream = batch_stream(X, Y, m=batch_size, shuffle=False, infinite=False)

        return self.evaluate_stream(stream, steps, metrics, verbose)

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
