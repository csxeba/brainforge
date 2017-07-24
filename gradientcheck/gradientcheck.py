import warnings

import numpy as np

from .raw_gradients import analytical_gradients, numerical_gradients
from .analyze_difference import analyze_difference_matrices, get_results


class GradientCheck:

    def __init__(self, network, epsilon=1e-5, display=True):
        if network.age <= 1:
            warnings.warn(
                "\nPerforming gradient check on an untrained neural network!",
                RuntimeWarning
            )
        self.net = network
        self.eps = epsilon
        self.dsp = display

    def run(self, X, Y):
        norm = np.linalg.norm
        analytic = analytical_gradients(self, X, Y)
        numeric = numerical_gradients(self, X, Y)
        diff = analytic - numeric
        relative_error = norm(diff) / max(norm(numeric), norm(analytic))
        passed = get_results(relative_error)

        if self.dsp and not passed:
            analyze_difference_matrices(self, diff)

        return passed
