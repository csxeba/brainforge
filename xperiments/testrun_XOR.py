import numpy as np

from brainforge import BackpropNetwork
from brainforge.architecture import DenseLayer
from optimization.gradient_descent import SGD as Opt

from matplotlib import pyplot as plt


BATCH_SIZE = 20


def input_stream(m):
    stimuli = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    while 1:
        arg = np.random.randint(len(stimuli), size=m)
        yield stimuli[arg], targets[arg]


net = BackpropNetwork(input_shape=(2,), layers=[
    DenseLayer(4, activation="sigmoid"),
    DenseLayer(2, activation="softmax")
])
net.finalize(cost="xent", optimizer=Opt(net.nparams, eta=1.))

datagen = input_stream(BATCH_SIZE)
validation = next(input_stream(100))

costs = net.fit_generator(datagen, lessons_per_epoch=20*100, epochs=30, monitor=["acc"],
                          validation=validation, verbose=0)

print("Did {} updates".format(len(costs)))
Xs = np.arange(1, len(costs)+1)
plt.scatter(Xs, costs, c="b", marker=".", alpha=0.1)
plt.ylim([0, 1])
plt.show()
