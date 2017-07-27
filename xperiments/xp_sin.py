import numpy as np

from matplotlib import pyplot as plt

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer


X = np.linspace(-6., 6., 100)[:, None]
Y = np.sin(X)

net = BackpropNetwork([DenseLayer(10, activation="tanh"),
                       DenseLayer(10, activation="tanh"),
                       DenseLayer(1, activation="linear")],
                      input_shape=1, optimizer="adagrad")

costs = net.fit(X, Y, batch_size=50, epochs=1000, verbose=0)
print("FINAL COST:", costs[-1])
plt.plot(X, Y, "b--")
plt.plot(X, net.predict(X), "r-")
plt.show()
