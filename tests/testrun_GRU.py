from brainforge import Network
from brainforge.layers import GRU, DenseLayer

from csxdata import Sequence, roots


frame = Sequence(roots["txt"] + "petofi.txt", timestep=5, n_gram=1, cross_val=0.01)

model = Network(frame.neurons_required[0], layers=(
    GRU(30, activation="tanh", return_seq=True),
    GRU(10, activation="tanh", return_seq=False),
    DenseLayer(frame.neurons_required[1], activation="softmax", trainable=True)
), name="TestGRU")
model.finalize("xent", optimizer="adam")

model.describe(1)

model.fit(*frame.table("learning", m=10), batch_size=10, epochs=1, verbose=0)
model.gradient_check(*frame.table("testing", m=30))

model.fit(*frame.table("learning"), monitor=["acc"], validation=frame.table("testing"))
