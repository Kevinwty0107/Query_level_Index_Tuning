from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np

epsilon = 0.00001


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def standardize(data):
    """
    Zero mean unit variance peprocessing, e.g. for states and rewards.
    Args:
        data:

    Returns:

    """

    return (data - np.mean(data)) / (np.std(data) + epsilon)