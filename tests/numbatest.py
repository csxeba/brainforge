"""Testing numba"""

import time

import numba
import numpy as np

from csxnet.util import white


@numba.jit
def mse(a: np.ndarray, y: np.ndarray):
    return np.sum((a - y)**2.).astype(float)


@numba.jit
def mse_p(a, y):
    return a - y


@numba.jit
def forward(x, w, b):
    z = x.dot(w) + b
    return 1. / (1. + np.exp(-z))


@numba.jit
def backward(e, w, x, a):
    e *= (a * (1. - a))
    nabla_b = np.sum(e, axis=0)
    nabla_w = x.T.dot(e)
    return e.dot(w.T), nabla_w, nabla_b


@numba.jit
def sgd(param, dparam, eta):
    return param - (dparam * eta)


@numba.jit
def epoch(Xs, Ys, Ws, bs, ETA):
    args = np.arange(Xs.shape[0])
    np.random.shuffle(args)
    Xs, Ys = Xs[args], Ys[args]

    As = [Xs]
    for w, b in zip(Ws, bs):
        As.append(forward(As[-1], w, b))

    cost = mse(As[-1], Ys)
    grads = [[np.array([])], [np.array()]]
    e = mse_p(As[-1], Ys)
    for w, x, a in zip(Ws[-1::-1], As[-2::-1], As[-1:0:-1]):
        e, nw, nb = backward(e, w, x, a)
        grads.append([nw, nb])
    grads = grads[-1:0:-1]

    eta = ETA / len(Xs)

    for i in range(len(Ws)):
        nw, nb = grads[i]
        Ws[i] = sgd(Ws[i], nw, eta)
        bs[i] = sgd(bs[i], nb, eta)

    return cost


def century(Xs: np.ndarray, Ys: np.ndarray, Ws: list, bs: list, ETA: float):
    costs = [0.0]
    for e in range(1000):
        costs.append(epoch(Xs, Ys, Ws, bs, ETA))
    return np.mean(costs[1:])


def learn(centuries, Xs, Ys, Ws, bs, ETA):
    start = time.time()
    for c in range(1, centuries+1):
        cost = century(Xs, Ys, Ws, bs, ETA)
        print("\rEpoch {0:>5}, Cost: {1:.4f}"
              .format(c*100, cost), end="")
    print()
    print("Time elapsed:", time.time()-start)


Wh = white(2, 10)
bh = np.zeros((10,))
Wo = white(10, 1)
bo = np.zeros((1,))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

learn(100, X, Y, [Wh, Wo], [bh, bo], ETA=1)
