from .gradient_descent import *
from .adaptive_gd import *

optimizers = {k: v for k, v in locals().items() if k not in
              ("np", "GradientDescent", "_SGD")}
