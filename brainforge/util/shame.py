def translate_architecture(arch):
    from ..layers.core import (
        InputLayer, Dense, Activation, Flatten, Reshape
    )

    from ..layers.fancy import (
        DropOut, Highway
    )

    from ..layers.recurrent import (
        Reservoir, LSTM, RLayer, ClockworkLayer
    )
    from ..layers.tensor import (
        ConvLayer, PoolLayer
    )

    dictionary = {
        "Inpu": InputLayer,
        "Dens": Dense,
        "Acti": Activation,
        "High": Highway,
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
