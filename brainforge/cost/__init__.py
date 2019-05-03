from ._costs import mean_squared_error, categorical_crossentropy, binary_crossentropy, hinge
from ._costs import mse, cxent, bxent
from ._costs import CostFunction

costs = {k: v for k, v in locals().items() if k[0] != "_" and k != "CostFunction"}
