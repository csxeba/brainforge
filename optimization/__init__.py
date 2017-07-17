from .gradient_descent import gdopt as _gd
from .adaptive_gd import agdopt as _agd

optimizers = dict()
optimizers.update(_gd)
optimizers.update(_agd)
