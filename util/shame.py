def translate_architecture(arch):
    from layers.core import (
        InputLayer, DenseLayer, Activation, Flatten, Reshape
    )

    from layers.fancy import (
        DropOut, HighwayLayer
    )

    from layers.recurrent import (
        Reservoir, LSTM, RLayer, ClockworkLayer
    )
    from layers.tensor import (
        ConvLayer, PoolLayer
    )

    dictionary = {
        "Inpu": InputLayer,
        "Dens": DenseLayer,
        "Acti": Activation,
        "High": HighwayLayer,
        "Drop": DropOut,
        "Flat": Flatten,
        "Resh": Reshape,
        "RLay": RLayer,
        "Cloc": ClockworkLayer,
        "LSTM": LSTM,
        "Echo": Reservoir,
        "MaxP": PoolLayer,
        "Conv": ConvLayer
    }

    return dictionary[arch[:4]]
