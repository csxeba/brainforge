from csxdata import etalon

from brainforge import Network
from brainforge.architectures.predictor import Predictor
from brainforge.layers import DenseLayer


frame = etalon()
tX, tY = frame.table("learning")

net = Network(frame.neurons_required[0], layers=(
    DenseLayer(12, activation="tanh"),
    DenseLayer(frame.neurons_required[1], activation="softmax")
), name="PredTestNet")
net.finalize("xent", "adam")

net.fit_csxdata(frame)
print("Network evaluation:")
print(net.evaluate(tX, tY, classify=True))
net.encapsulate(".")

pred = Predictor("./PredTestNet.cps")
print("Predictor evaluation:")
print(pred.evaluate(tX, tY, net.cost, classify=True))
