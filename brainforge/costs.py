from brainforge import backend as xp


class CostFunction:

    def __call__(self, outputs, targets): pass

    def __str__(self): return ""

    @staticmethod
    def simplified_derivative(outputs, targets):
        return outputs - targets


class MSE(CostFunction):

    def __call__(self, outputs, targets):
        return ((outputs - targets)**2).sum() / 2

    @staticmethod
    def derivative(outputs, targets):
        return outputs - targets


class CXent(CostFunction):

    def __call__(self, outputs, targets):
        return -(targets * xp.log(outputs)).sum()

    def derivative(self, outputs, targets):
        ...


class BXent(CostFunction):

    def __call__(self, outputs, targets):
        return

    def derivative(self, outputs, targets):
        enum = targets - outputs
        denom = (outputs - 1) * outputs
        return enum / denom


class Hinge(CostFunction):

    def __call__(self, outputs, targets):
        return (xp.maximum(0, 1 - targets * outputs)).sum()

    def __str__(self):
        return "Hinge"

    @staticmethod
    def derivative(outputs, targets):
        """
        Using subderivatives,
        d/da = -y whenever output > 0
        """
        out = -targets
        out[outputs > 1] = 0
        return out

    def simplified_derivative(self, outputs, targets):
        return NotImplemented


xent = CXent()
cxent = xent
bxent = BXent()
hinge = Hinge()
mse = MSE()

cost_functions = {
    "xent": xent,
    "cxent": cxent,
    "bxent": bxent,
    "hinge": hinge,
    "mse": mse
}


def get(function: str):
    return cost_functions[function]
