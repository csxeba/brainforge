import numpy as np

from .raw_gradients import analytical_gradients, numerical_gradients
from .analyze_difference import analyze_difference_matrices, get_results
from ..learner import Backpropagation


def run(network: Backpropagation, X=None, Y=None, epsilon=1e-5, throw=False, display=True):
    if X is None:
        X = np.random.normal(scale=0.1, size=network.input_shape)
    if Y is None:
        Y = np.random.normal(scale=0.1, size=network.output_shape)
    norm = np.linalg.norm
    analytic = analytical_gradients(network, X, Y)
    numeric = numerical_gradients(network, X, Y, epsilon)
    diff = analytic - numeric
    relative_error = norm(diff) / max(norm(numeric), norm(analytic))
    passed = get_results(relative_error)

    if display and not passed:
        analyze_difference_matrices(network, diff)
    if throw and not passed:
        raise RuntimeError("Gradient Check failed")
    return passed
