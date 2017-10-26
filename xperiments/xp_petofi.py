import os
from csxdata import LazyText

from brainforge import BackpropNetwork
from brainforge.layers import LSTM, DenseLayer
from brainforge.optimization import RMSprop

from keras.models import Sequential
from keras.layers import LSTM as kLSTM, Dense as kDense

data = LazyText(os.path.expanduser("~/tmp/RIS.txt.txt"), n_gram=1, timestep=5)
inshape, outshape = data.neurons_required
net = BackpropNetwork(input_shape=inshape, layerstack=[
    LSTM(60, activation="tanh"),
    DenseLayer(60, activation="tanh"),
    DenseLayer(outshape, activation="softmax")
], cost="xent", optimizer=RMSprop(eta=0.01))

net.fit_generator(data.batchgen(20), lessons_per_epoch=data.N)

keras = Sequential([
    kLSTM(60, input_shape=inshape, activation="tanh"),
    kDense(60, activation="tanh"),
    kDense(outshape, activation="softmax")
])
keras.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
keras.fit_generator(data.batchgen(20), steps_per_epoch=data.N, epochs=30)
