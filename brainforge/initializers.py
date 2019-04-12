from brainforge import backend as xp


def normal(*shape, mean=0., stdev=1.):
    return xp.random.normal(mean, stdev, shape)


def normal_like(array: xp.ndarray, mean=0., stdev=1.):
    return normal(*array.shape, mean, stdev)


def glorot_normal(*shape):

    if len(shape) == 2:
        num_input, num_output = shape
    elif len(shape) == 4:
        fy, fx, fc, nf = shape
        num_input, num_output = nf*fy*fx + fc*fy*fx
    else:
        assert False

    return xp.random.randn(*shape) * xp.sqrt(2. / float(num_input + num_output))


initializers = {"normal": normal, "glorot_normal": glorot_normal}
