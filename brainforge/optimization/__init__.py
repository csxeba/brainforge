from .gradient_descent import gdopt as _gd, SGD, Momentum, GradientDescent
from .adaptive_gd import agdopt as _agd, Adagrad, RMSprop, Adam

optimizers = dict()
optimizers.update(_gd)
optimizers.update(_agd)
