def translate_architecture(arch):
    from layers.core import (
        InputLayer, DenseLayer, Activation,
        Flatten, Reshape)
    from layers.fancy import DropOut
    from layers.fancy import HighwayLayer
    from layers.recurrent import Reservoir
    from layers.recurrent import LSTM
    from layers.recurrent import RLayer
    from layers.tensor import ConvLayer
    from layers.tensor import PoolLayer

    dictionary = {
        "Inpu": InputLayer,
        "Dens": DenseLayer,
        "Acti": Activation,
        "High": HighwayLayer,
        "Drop": DropOut,
        "Flat": Flatten,
        "Resh": Reshape,
        "RLay": RLayer,
        "LSTM": LSTM,
        "Echo": Reservoir,
        "MaxP": PoolLayer,
        "Conv": ConvLayer
    }

    return dictionary[arch[:4]]
