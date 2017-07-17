def translate_architecture(arch):
    from ..architecture.core import (
        InputLayer, DenseLayer, Activation, Flatten, Reshape
    )

    from ..architecture.fancy import (
        DropOut, HighwayLayer
    )

    from ..architecture.recurrent import (
        Reservoir, LSTM, RLayer, ClockworkLayer
    )
    from ..architecture.tensor import (
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
        "Rese": Reservoir,
        "Pool": PoolLayer,
        "Conv": ConvLayer
    }

    return dictionary[arch[:4]]
