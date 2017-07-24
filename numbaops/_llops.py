import numba as nb


@nb.jit(nopython=True)
def dense_forward(X, W, b):
    return X.dot(W) + b


@nb.jit(nopython=True)
def dense_backward(X, E, W):
    gW = X.T.dot(E)
    gb = E.sum(axis=0)
    gX = E.dot(W.T)
    return gW, gb, gX
