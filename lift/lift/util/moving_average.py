import numpy as np
from collections import deque


class MovingAverage(object):
    """
    Simple moving average object.

    If space is costly, one could
    implement an incremental version.
    """

    def __init__(self, size=1000):
        self.data = deque(maxlen=size)

    def get_incremental_average(self, item):
        """
        Add item to data, compute average.

        Args:
            item (Union[float, int]):

        Returns:
            Mean, std
        """
        self.data.append(item)

        list_data = list(self.data)

        return np.mean(list_data), np.std(list_data)