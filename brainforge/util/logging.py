from collections import defaultdict
from statistics import mean


class MetricLogs:

    def __init__(self, fields, max_steps=-1):
        self._metrics = defaultdict(list)
        self._max_steps = max_steps
        self._step = 0
        self.reduced = False

    @classmethod
    def from_metric_list(cls, max_steps, fields=(), metric_list=()):
        fields = list(fields) + [str(metric).lower() for metric in metric_list]
        return cls(fields, max_steps)

    def record(self, data: dict):
        step = 1
        for key, value in data.items():
            if isinstance(value, list):
                self._metrics[key].extend(value)
                step = len(value)
            else:
                self._metrics[key].append(value)
        self._step += step

    def update(self, data: "MetricLogs"):
        self.record(data._metrics)

    def __getitem__(self, item):
        return self._metrics[item]

    def log(self, prefix="", suffix="", **print_kwargs):
        log_str = []
        if self._max_steps > 0:
            log_str.append("Progress: {:>6.1%} ".format(self._step / self._max_steps))
        means = self.mean()
        log_str += ["{}: {:.4f}".format(metric, metric_values) for metric, metric_values in means.items()]
        print(prefix + " ".join(log_str) + suffix, **print_kwargs)

    def mean(self):
        if self.reduced:
            return self._metrics
        return {k: mean(v) for k, v in self._metrics.items()}

    def reduce_mean(self):
        self._metrics = {k: mean(v) for k, v in self._metrics.items()}
        self.reduced = True
