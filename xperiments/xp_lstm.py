import numpy as np

from matplotlib import pyplot as plt

from brainforge.layers import LSTM
from brainforge.layers.core import InputLayer
from brainforge.atomic.recurrent_op import LSTMOp

np.random.seed(1337)

BSZE = 20
TIME = 5
DDIM = 10
NEUR = 15

inp = InputLayer()
inp.connect(None, (TIME, DDIM))
lr = LSTM(NEUR, activation="tanh", return_seq=True)
lr.connect(inp, inp.outshape)

X = np.random.randn(BSZE, TIME, DDIM)
W = np.random.randn(NEUR+DDIM, NEUR*4)
b = np.random.randn(NEUR*4)
E = np.random.randn(BSZE, TIME, NEUR)

lr.weights = W
lr.biases = b
op = LSTMOp("tanh")

lrO = lr.feedforward(X)
opO, cache = op.forward(X.transpose((1, 0, 2)), W, b)
opO = opO.transpose((1, 0, 2))

d = (lrO - opO).reshape(20, 5*15)

plt.imshow(np.abs(d), cmap="hot")
plt.show()

assert np.allclose(lrO, opO), f"Diff ~ {np.abs(opO - lrO).sum():.4f}"
