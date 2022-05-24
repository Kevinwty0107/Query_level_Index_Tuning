import numpy as np


class ObjectiveScaler(object):

    def __init__(self, means, stddevs=None, ns=None, combine_estimates = False, 
            verbose = False):
        self.verbose = verbose
        if combine_estimates and not ns:
            raise RuntimeError('If combining estimates of variance,' +
                    'must provide the sample sizes')
        if combine_estimates:
            self._combine_mean(means, ns)
            self._combine_stddev(stddevs, ns)
        else:
            self.mean = np.mean(means)
            self.stddev = np.std(means)

    def _compute_mean(self, means, ns):
        self.mean = np.dot(means, ns)
        self.mean = self.mean / np.sum(ns)

    def _compute_stddev(self, stddevs, ns):
        ns = ns - 1
        stddevs = stddevs * stddevs
        self.stddev = np.dot(stddevs, ns)
        self.stddev = np.sqrt(self.stddev / np.sum(ns))

    def scale(self, x):
        if self.verbose:
            print('Mean: {}'.format(self.mean))
            print('Stddev: {}'.format(self.stddev))
        return (x - self.mean) / self.stddev
