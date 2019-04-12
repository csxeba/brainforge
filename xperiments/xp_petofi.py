import os
from csxdata import LazyText

from brainforge import BackpropNetwork
from brainforge.layers import LSTM, Linear
from brainforge.optimization import RMSprop

from keras.models import Sequential
from keras.layers import LSTM as kLSTM, Dense as kDense, BatchNormalization

data = LazyText(os.path.expanduser("~/Prog/data/txt/scripts.txt"), n_gram=1, timestep=10)
inshape, outshape = data.neurons_required


def run_brainforge():
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        LSTM(60, activation="tanh"),
        Linear(60, activation="tanh"),
        Linear(outshape, activation="softmax")
    ], cost="xent", optimizer=RMSprop(eta=0.01))

    net.fit_generator(data.batchgen(20), lessons_per_epoch=data.N)


def run_keras():
    keras = Sequential([
        kLSTM(120, input_shape=inshape, activation="relu"), BatchNormalization(),
        kDense(60, activation="relu"), BatchNormalization(),
        kDense(outshape, activation="softmax")
    ])
    keras.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    keras.fit_generator(data.batchgen(50), steps_per_epoch=data.N, epochs=30)


if __name__ == '__main__':
    run_keras()
