from brainforge import backend as xp


class Regularizer:

    def __init__(self, coef=0.1):
        self.coef = coef

    def __call__(self, param):
        raise NotImplementedError

    def derivative(self, param):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class L1Norm(Regularizer):

    def __call__(self, param):
        return self.coef * xp.abs(param).sum()

    def derivative(self, param):
        return self.coef * xp.sign(param)


class L2Norm(Regularizer):

    def __call__(self, param):
        return 0.5 * self.coef * (param**2.).sum()

    def derivative(self, param):
        return self.coef * param
