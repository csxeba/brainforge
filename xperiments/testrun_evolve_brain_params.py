from csxdata import CData, roots

from brainforge import NeuroEvolution
from brainforge.architecture import DenseLayer


data = CData(roots["misc"] + "mnist.pkl.gz", cross_val=10000, fold=False,
             floatX="float64")
net = NeuroEvolution(data.neurons_required[0], layers=(
    DenseLayer(30, activation="sigmoid"),
    DenseLayer(data.neurons_required[1], activation="sigmoid")
))
net.finalize("mse", population_size=30, on_accuracy=False)

print("Initial acc:", net.evaluate(*data.table("testing"))[1])
net.fit(*data.table("learning", m=10000), batch_size=500, validation=data.table("testing"), monitor=["acc"])
