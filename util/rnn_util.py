import numpy as np


def speak_to_me(model, csxframe, stochastic=False, ngrams=50):
    pred = csxframe.primer()
    human = csxframe.translate(pred)
    chain = "[{}]".format("".join(human))
    for _ in range(ngrams):
        inputs = pred[:, -(csxframe.timestep - 1):, :]
        nextpred = model.predict(inputs)
        pred = np.column_stack((pred, nextpred.reshape(1, *nextpred.shape)))
        human = csxframe.translate(nextpred, use_proba=stochastic)
        chain += "".join(human)
    return chain


def keras_speak(net, dat, stochastic=False, ngrams=50):
    pred = dat.primer()
    human = dat.translate(pred)
    chain = "[{}]".format("".join(human))
    for _ in range(ngrams):
        inputs = pred[:, -(dat.timestep - 1):, :]
        nextpred = net.predict(inputs)
        pred = np.column_stack((pred, nextpred.reshape(1, *nextpred.shape)))
        human = dat.translate(nextpred, use_proba=stochastic)
        chain += "".join(human)
    return chain
