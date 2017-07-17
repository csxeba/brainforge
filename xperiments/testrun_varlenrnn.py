from brainforge import BackpropNetwork
from brainforge.architecture import RLayer, DenseLayer

from csxdata import WordSequence, roots

pet = WordSequence(roots["txt"] + "petofi.txt", cross_val=0,
                   lower=True, dehun=True, decimal=True)

net = BackpropNetwork(input_shape=pet.neurons_required[0], layers=(
    RLayer(30, "relu"),
    DenseLayer(60, activation="tanh"),
    DenseLayer(pet.neurons_required[1], activation="softmax")
)).finalize("xent", "adam")
net.fit(*pet.table("learning"))
