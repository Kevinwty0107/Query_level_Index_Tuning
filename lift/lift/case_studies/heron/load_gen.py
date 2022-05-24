from random import randint
import numpy as np
import logging


class LoadGenerator(object):
    def __init__(self, load_config, parallelism):
        self.logger = logging.getLogger(__name__)
        self.parallelism = parallelism[load_config['component']]
        self.periods = load_config['load_generator_args']['periods']
        self.file_name = load_config['load_generator_args']['save_file']
        self.component = load_config['component']
        # for replaying
        # self.memory = np.zeros((self.periods,), dtype=np.int32)

    def loads(self):
        raise NotImplementedError

    def replay(self):
        """ 
        Replays a previous load to allow direct comparison of 
        different methods
        """
        self.restore()
        for i in range(self.periods):
            yield self.memory[i]

    def serialise(self):
        """
        Saves the memory to a file.
        """
        np.save(self.file_name, self.memory)

    def restore(self):
        """
        Restores the memory
        """
        self.memory = np.load(self.file_name)


class AlternatingFixedLoadGenerator(LoadGenerator):

    def __init__(self, load_config, parallelism):
        super(AlternatingFixedLoadGenerator, self).__init__(load_config,
                                                            parallelism)
        self.period_length = load_config['load_generator_args']['period_length']
        self.low = load_config['load_generator_args']['low']
        self.high = load_config['load_generator_args']['high']
        self.memory = np.zeros((self.periods * self.period_length,),
                               dtype=np.int32)

    def loads(self):
        for i in range(self.periods):
            if i % 2 == 0:
                par = self.high
            else:
                par = self.low
            for j in range(self.period_length):
                self.memory[i * self.period_length + j] = par
                yield {self.component: par}


class UpDownLoadGenerator(LoadGenerator):

    def __init__(self, load_config, parallelism):
        super(UpDownLoadGenerator, self).__init__(load_config, parallelism)
        self.high = load_config['load_generator_args']['high']
        self.low = load_config['load_generator_args']['low']
        self.memory = []
        self.up_length = load_config['load_generator_args']['start_length']
        self.epoch_length = load_config['load_generator_args']['epoch_length']

    def loads(self):
        for i in range(self.up_length):
            self.memory.append(self.high)
            yield {self.component: self.high}
        down_length = self.epoch_length - self.up_length
        for i in range(down_length):
            self.memory.append(self.low)
            yield {self.component: self.low}


class DownUpLoadGenerator(LoadGenerator):

    def __init__(self, load_config, parallelism):
        super(DownUpLoadGenerator, self).__init__(load_config, parallelism)
        self.high = load_config['load_generator_args']['high']
        self.low = load_config['load_generator_args']['low']
        self.memory = []
        self.down_length = load_config['load_generator_args']['start_length']
        self.epoch_length = load_config['load_generator_args']['epoch_length']

    def loads(self):
        for i in range(self.down_length):
            self.memory.append(self.low)
            yield {self.component: self.low}
        up_length = self.epoch_length - self.down_length
        for i in range(up_length):
            self.memory.append(self.high)
            yield {self.component: self.high}


class AlternatingLoadGenerator(LoadGenerator):

    def __init__(self, load_config, parallelism):
        super(AlternatingLoadGenerator, self).__init__(load_config,
                                                       parallelism)
        self.max_period_length = \
            load_config['load_generator_args']['max_period_length']
        self.low = load_config['load_generator_args']['low']
        self.high = load_config['load_generator_args']['high']
        self.memory = []

    def loads(self):
        for i in range(self.periods):
            if i % 2 == 0:
                par = self.low
            else:
                par = self.high
            period_length = randint(1, self.max_period_length)
            for j in range(period_length):
                self.memory.append(par)
                yield {self.component: par}
