from brainforge import Network
from brainforge.layers import GRU, DenseLayer

from csxdata import Sequence, roots


frame = Sequence(roots["txt"] + "petofi.txt", timestep=6)

model = Network(frame.neurons_required[0], layers=(
    GRU(300, activation="relu", return_seq=True),
    GRU(180, activation="relu", return_seq=False),
    DenseLayer(120, activation="tanh"),
    DenseLayer(frame.neurons_required[1], activation="softmax")
))
model.finalize("xent", optimizer="adam")

model.fit(*frame.table("learning", m=10), batch_size=10, epochs=1, verbose=0)
model.gradient_check(*frame.table("testing", m=30))

model.fit(*frame.table("learning"), monitor=["acc"], validation=frame.table("testing"))
