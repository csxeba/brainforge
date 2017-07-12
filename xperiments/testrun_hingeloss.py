import numpy as np

from brainforge import Network
from brainforge.layers import DenseLayer

from csxdata import CData, roots

frame = CData(roots["misc"] + "mnist.pkl.gz", fold=False)

model = Network(input_shape=frame.neurons_required[0], layers=(
    DenseLayer(30, activation="tanh"),
    DenseLayer(frame.neurons_required[-1], activation="linear")
))
model.finalize(cost="hinge")

Xs, Ys = frame.table("learning", m=5)
Ya = -np.ones_like(Ys) + 2*Ys
assert Ya.min() == -1. and Ya.max() == 1.

model.describe(1)
print("Grad check on  0 - 1 Y")
model.gradient_check(Xs, Ys)
print("Grad check on -1 - 1 Y")
model.gradient_check(Xs, Ya)

model.fit(*frame.table("learning"), batch_size=100)
