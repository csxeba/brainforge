import numpy as np

from . import act_fns

sigmoid = act_fns["sigmoid"]


def dense_op(prm):
    w, b = prm[-1]
    act = act_fns[prm[-2]]
    return lambda x: act(x @ w + b)


def activation_op(prm):
    act = act_fns[prm[-1]]
    return lambda x: act(x)


def dropout_op(prm):
    return lambda x: x * prm[-1]


def highway_op(prm):

    def op(x, w, b, act):
        x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        gates = x @ w + b
        gates[:, :neu] = act(gates[:, :neu])
        gates[:, neu:] = sigmoid(gates[:, neu:])
        h, t, c = np.split(gates, 3, axis=1)
        return (h * t + x * c).reshape(x.shape)

    weights, biases = prm[-1]
    neu = weights.shape[0]
    activation = act_fns[prm[-2]]
    return lambda stim: op(stim, weights, biases, activation)


def conv_valid_op(prm):

    def op(x, F):
        im, ic, iy, ix = x.shape
        nf, fc, fy, fx = F.shape
        recfield_size = fx * fy * fc
        oy, ox = (iy - fy) + 1, (ix - fx) + 1
        rfields = np.zeros((im, oy * ox, recfield_size))

        if fc != ic:
            err = "Supplied filter (F) is incompatible with supplied input! (X)\n" \
                  + "input depth: {} != {} :filter depth".format(ic, fc)
            raise ValueError(err)

        for i, pic in enumerate(x):
            for sy in range(oy):
                for sx in range(ox):
                    rfields[i][sy * ox + sx] = pic[:, sy:sy + fy, sx:sx + fx].ravel()

        return (np.matmul(rfields, F.reshape(nf, recfield_size).T)
                .transpose(0, 2, 1).reshape(im, nf, oy, ox))

    filters, biases = prm[-1]
    activation = act_fns[prm[-2]]

    return lambda x: activation(op(x, filters))


ops = {k: v for k, v in locals().items() if k[:-3] == "_op"}
