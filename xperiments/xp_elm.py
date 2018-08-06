import numpy as np

from csxdata.utilities.loader import pull_mnist_data

from brainforge import LayerStack
from brainforge.layers import DenseLayer
from brainforge.learner.elm import ExtremeLearningMachine


lX, lY, tX, tY = pull_mnist_data()

layers = LayerStack(input_shape=(784,), layers=[
    DenseLayer(60, activation="tanh", trainable=False),
    DenseLayer(10, activation="linear", trainable=True)
])

elm = ExtremeLearningMachine(layers, cost="mse")
elm.learn_batch(tX, tY)
pred = elm.predict(tX)
print("ELM cost:", elm.cost(pred, tY))
print("ELM acc :", np.mean(tY.argmax(axis=0) == pred.argmax(axis=0)))
pass
