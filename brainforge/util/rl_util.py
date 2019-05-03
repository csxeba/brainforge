import numpy as np


def discount_rewards(rwd, gamma=0.99):
    """
    Compute the discounted reward backwards in time
    """
    discounted_r = np.zeros_like(rwd)
    running_add = rwd[-1]
    for t, r in enumerate(rwd[::-1]):
        running_add += gamma * r
        discounted_r[t] = running_add
    discounted_r[0] = rwd[-1]
    return discounted_r[::-1]
