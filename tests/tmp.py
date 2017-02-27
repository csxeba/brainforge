import numpy as np

from brainforge import Network
from brainforge.layers import DenseLayer

def input_stream(m=20):
    Xs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Ys = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    while 1:
        arg = np.random.randint(len(Xs), size=m)
        yield Xs[arg], Ys[arg]

net = Network(input_shape=(2,), layers=[
    DenseLayer(12, activation="sigmoid"),
    DenseLayer(2, activation="sigmoid")
])
net.finalize(cost="xent", optimizer="adam")

datagen = input_stream(1000)
valid_stream = input_stream(100)

for epoch, (X, Y) in enumerate(datagen, start=1):
    print("Epoch", epoch+1)
    net.epoch(X, Y, batch_size=20, monitor=["acc"],
              validation=next(valid_stream), verbose=1)
    if epoch == 30:
        break
