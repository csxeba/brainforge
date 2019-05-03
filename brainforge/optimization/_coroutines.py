import numpy as np


def sgd(nparams, eta, *args):
    nabla = np.zeros((nparams,))
    while 1:
        m, W, gW = (yield nabla)
        nabla = (W - gW) * (eta / m)


def momentum(nparams, eta, mu, *args):
    velocity = np.zeros((nparams,))
    while 1:
        m, W, gW = (yield velocity)
        nabla = gW * (eta/m)
        velocity *= mu
        velocity += nabla


def nesterov(nparams, eta, mu, *args):
    update = np.zeros((nparams,))
    velocity = np.zeros_like(update)
    memory = np.zeros_like(velocity)
    while 1:
        m, W, gW = (yield update)
        nabla = gW * (eta/m)
        update = memory - velocity + nabla
        velocity *= mu
        velocity += nabla


def adagrad(nparams, eta, epsilon, *args):
    update = np.zeros((nparams,))
    memory = np.zeros_like(update)
    while 1:
        m, W, gW = (yield update)
        nabla = gW / m
        memory += (nabla**2)
        update = (eta / np.sqrt(memory + epsilon)) * nabla


def rmsprop(nparams, eta, decay, epsilon, *args):
    update = np.zeros((nparams,))
    memory = np.zeros_like(update)
    while 1:
        m, W, gW = (yield update)
        nabla = gW / m
        memory *= decay
        memory += (1. - decay) * (nabla**2)
        update = (eta / np.sqrt(memory + epsilon)) * nabla


def adam(nparams, eta, decay_velo, decam_memo, epsilon, *args):
    update = np.zeros((nparams,))
    memory = np.zeros_like(update)
    velocity = np.zeros_like(update)
    while 1:
        m, W, gW = (yield update)
        lr = eta / m
        velocity *= decay_velo
        memory *= decam_memo
        velocity += (1. - decay_velo) * gW
        memory += (1. - decam_memo) * (gW**2)
        update = (lr / np.sqrt(memory + epsilon)) * velocity
