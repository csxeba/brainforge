import abc

import numpy as np


def l1term(eta, lmbd, N):
    return (eta * lmbd) / N


def l2term(eta, lmbd, N):
    return 1 - ((eta * lmbd) / N)


def numerical_gradients(network, X, y, epsilon=1e-5):
    ws = network.get_weights(unfold=True)
    numgrads = np.zeros_like(ws)
    perturb = np.copy(numgrads)

    nparams = ws.size
    print("Calculating numerical gradients...")
    for i in range(nparams):
        print("\r{0:>{1}} / {2:<}".format(i+1, len(str(nparams)), nparams), end=" ")
        perturb[i] += epsilon

        network.set_weights(ws + perturb, fold=True)
        pred1 = network.prediction(X)
        cost1 = network.cost(pred1, y)
        network.set_weights(ws - perturb, fold=True)
        pred2 = network.prediction(X)
        cost2 = network.cost(pred2, y)

        numgrads[i] = (cost1 - cost2)
        perturb[i] = 0.0

    numgrads /= (2 * epsilon)
    network.set_weights(ws, fold=True)

    print("Done!")

    return numgrads


def analytical_gradients(network, X, y):
    anagrads = np.zeros((network.nparams,))
    network.prediction(X)
    network.backpropagation(y)

    start = 0
    for layer in network.layers:
        if not layer.trainable:
            continue
        end = start + layer.nparams
        anagrads[start:end] = layer.gradients
        start = end

    return anagrads


def gradient_check(network, X, y, epsilon=1e-5, display=True, verbose=1):

    def fold_difference_matrices(dvec):
        diffs = []
        start = 0
        for layer in network.layers[1:]:
            if not layer.trainable:
                continue
            iweight = start + layer.weights.size
            ibias = iweight + layer.biases.size
            wshape = [sh for sh in layer.weights.shape if sh > 1]
            bshape = [sh for sh in layer.biases.shape if sh > 1]
            diffs.append(dvec[start:iweight].reshape(wshape))
            diffs.append(dvec[iweight:ibias].reshape(bshape))
            start = ibias
        return diffs

    def analyze_difference_matrices(dvec):
        dmats = fold_difference_matrices(np.abs(dvec))
        for i, d in enumerate(dmats):
            print("Sum of difference matrix no {0}: {1:.4e}".format(i, d.sum()))
            display_matrices(d)

    def display_matrices(mats):
        from matplotlib import pyplot

        if mats.ndim > 2:
            for mat in mats:
                display_matrices(mat)
        else:
            pyplot.matshow(np.atleast_2d(mats))
            pyplot.show()

    def get_results(er):
        if relative_error < 1e-7:
            errcode = 0
        elif relative_error < 1e-5:
            errcode = 1
        elif relative_error < 1e-3:
            errcode = 2
        else:
            errcode = 3

        if verbose:
            print("Result of gradient check:")
            print(["Gradient check passed, error {} < 1e-7",
                   "Suspicious gradients, 1e-7 < error {} < 1e-5",
                   "Gradient check failed, 1e-5 < error {} < 1e-3",
                   "Fatal fail in gradient check, error {} > 1e-3"
                   ][errcode].format("({0:.1e})".format(er)))

        return True if errcode < 3 else False

    norm = np.linalg.norm
    analytic = analytical_gradients(network, X, y)
    numeric = numerical_gradients(network, X, y, epsilon=epsilon)
    diff = analytic - numeric
    relative_error = norm(diff) / max(norm(numeric), norm(analytic))

    passed = get_results(relative_error)

    if display and not passed:
        analyze_difference_matrices(diff)

    return passed


def white(*dims) -> np.ndarray:
    """Returns a white noise tensor"""
    return np.random.randn(*dims) / np.sqrt(dims[0] / 2.)


def white_like(array):
    return white(*array.shape)


def rtm(A):
    """Converts an ndarray to a 2d array (matrix) by keeping the first dimension as the rows
    and flattening all the other dimensions to columns"""
    if A.ndim == 2:
        return A
    A = np.atleast_2d(A)
    return A.reshape(A.shape[0], np.prod(A.shape[1:]))


class _CostFnBase(abc.ABC):

    def __call__(self, outputs, targets): pass

    def __str__(self): return ""

    @staticmethod
    def derivative(outputs, targets): pass


class _MSE(_CostFnBase):

    def __call__(self, outputs, targets):
        return 0.5 * np.linalg.norm(outputs - targets) ** 2

    @staticmethod
    def derivative(outputs, targets):
        return np.subtract(outputs, targets)

    def __str__(self):
        return "MSE"


class _Xent(_CostFnBase):

    def __call__(self, a: np.ndarray, y: np.ndarray):
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def derivative(outputs, targets):
        enum = np.subtract(targets, outputs)
        denom = np.subtract(outputs, 1.) * outputs
        d_xent = np.divide(enum, denom)
        return d_xent

    def __str__(self):
        return "Xent"


class _NLL(_CostFnBase):
    """Negative logstring-likelyhood cost function"""
    def __call__(self, outputs, targets=None):
        return np.negative(np.log(outputs))

    @staticmethod
    def derivative(outputs, targets=None):
        return np.reciprocal(outputs)

    def __str__(self):
        return "NLL"


class _Cost:
    @property
    def mse(self):
        return _MSE()

    @property
    def xent(self):
        return _Xent()

    @property
    def nll(self):
        return _NLL()

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        item = item.lower()
        d = {str(fn).lower(): fn for fn in (_NLL(), _MSE(), _Xent())}
        if item not in d:
            raise IndexError("Requested cost function is unsupported!")
        return d[item]


mse = _MSE()
xent = _Xent()
nll = _NLL()
cost_fns = _Cost()
