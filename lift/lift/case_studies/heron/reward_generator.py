import numpy as np

class RewardGenerator(object):

    def reward(self, data):
        raise NotImplementedError


class LinearTLRewardGenerator(RewardGenerator):

    def __init__(self, config):
        super(LinearTLRewardGenerator, self).__init__()
        assert ('alpha' in config and config['alpha'] >= 0.0 and \
                config['alpha'] <= 1.0)
        self.alpha = config['alpha']

    def reward(self, data):
        # need negative latency because want higher to be better
        return self.alpha * (-data['latency']) + (1.0 - self.alpha) * \
            data['throughput']


class ResourceUsageRewardGenerator(RewardGenerator):

    def __init__(self, config):
        super(ResourceUsageRewardGenerator, self).__init__()
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.max_instances = config['max_instances']
        assert (self.alpha >= 0.0 and self.beta >= 0.0 and self.alpha + 
                self.beta <= 1.0)

    def reward(self, data):
        # take sqrt of latency
        return self.alpha * (-data['latency']) + self.beta * data['throughput']\
                - (1.0 - self.alpha - self.beta) * (data['instances'] / \
                self.max_instances)


