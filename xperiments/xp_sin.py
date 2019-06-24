import numpy as np

from matplotlib import pyplot as plt

from brainforge.learner import BackpropNetwork
from brainforge.layers import DenseLayer

np.random.seed(1234)

rX = np.linspace(-6., 6., 200)[:, None]
rY = np.sin(rX)

arg = np.arange(len(rX))
np.random.shuffle(arg)
targ, varg = arg[:100], arg[100:]
targ.sort()
varg.sort()

tX, tY = rX[targ], rY[targ]
vX, vY = rX[varg], rY[varg]

tX += np.random.randn(*tX.shape) / np.sqrt(tX.size*0.25)

net = BackpropNetwork([DenseLayer(120, activation="tanh"),
                       DenseLayer(120, activation="tanh"),
                       DenseLayer(1, activation="linear")],
                      input_shape=1, optimizer="adam")

tpred = net.predict(tX)
vpred = net.predict(vX)
plt.ion()
plt.plot(tX, tY, "b--", alpha=0.5, label="Training data (noisy)")
plt.plot(rX, rY, "r--", alpha=0.5, label="Validation data (clean)")
plt.ylim(-2, 2)
plt.plot(rX, np.ones_like(rX), c="black", linestyle="--")
plt.plot(rX, -np.ones_like(rX), c="black", linestyle="--")
plt.plot(rX, np.zeros_like(rX), c="grey", linestyle="--")
tobj, = plt.plot(tX, tpred, "bo", markersize=3, alpha=0.5, label="Training pred")
vobj, = plt.plot(vX, vpred, "ro", markersize=3, alpha=0.5, label="Validation pred")
templ = "Batch: {:>5}, tMSE: {:>.4f}, vMSE: {:>.4f}"
t = plt.title(templ.format(0, 0., 0.))
plt.legend()
batchno = 1
while 1:
        tmetrics = net.learn_batch(tX, tY)
        tpred = net.predict(tX)
        vpred = net.predict(vX)
        vcost = net.cost(vpred, vY) / len(vpred)
        tobj.set_data(tX, tpred)
        vobj.set_data(vX, vpred)
        plt.pause(0.1)
        t.set_text(templ.format(batchno, tmetrics["cost"], vcost))
        batchno += 1
