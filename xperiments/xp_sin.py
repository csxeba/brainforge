import numpy as np

from matplotlib import pyplot as plt

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer

np.random.seed(1234)

X = np.linspace(-6., 6., 100)[:, None]
Y = np.sqrt(np.sin(X) + 1)

net = BackpropNetwork([DenseLayer(120, activation="relu"),
                       DenseLayer(120, activation="relu"),
                       DenseLayer(40, activation="relu"),
                       DenseLayer(1, activation="linear")],
                      input_shape=1, optimizer="adam")

pred = net.predict(X)
plt.ion()
plt.plot(X, Y, "b--")
plt.ylim(-2, 2)
plt.plot(X, np.ones_like(X), c="black", linestyle="--")
plt.plot(X, -np.ones_like(X), c="black", linestyle="--")
plt.plot(X, np.zeros_like(X), c="grey", linestyle="--")
obj, = plt.plot(X, pred, "r-", linewidth=2)
batchno = 1
while 1:
        cost = net.learn_batch(X, Y)
        pred = net.predict(X)
        obj.set_data(X, pred)
        plt.pause(0.01)
        plt.title(f"Batch: {batchno:>5}, MSE: {cost:>.4f}")
        batchno += 1
