import random
import numpy as np


class LoadGenerator(object):
    def __init__(self, load_config, parallelism):
        self.parallelism = parallelism[load_config['component']]
        self.periods = load_config['load_generator_args']['periods']
        self.file_name = load_config['load_generator_args']['save_file']
        # for replaying
        self.memory = np.zeros((self.periods,), dtype=np.int32)

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

class AlternatingLoadGenerator(LoadGenerator):

    def __init__(self, load_config, parallelism):
        super(AlternatingLoadGenerator, self).__init__(load_config, parallelism)
        self.high = load_config['load_generator_args']['high']
    
    def get_memory(self):
        return self.memory

    def loads(self):
        for i in range(self.periods):
            if i % 2 == 0: 
                self.memory[i] = self.high
                yield self.high
            else:
                self.memory[i] = self.parallelism
                yield self.parallelism
        self.serialise()


class ConstantLoadGenerator(LoadGenerator):
    def __init__(self, load_config, parallelism):
        super(ConstantLoadGenerator, self).__init__(load_config, parallelism)
        self.constant = load_config['load_generator_args']['constant']

    def loads(self):
        for i in range(self.periods):
            self.memory[i] = self.constant
            yield self.constant
        self.serialise()

class BrownianLoadGenerator(LoadGenerator):
    def __init__(self, load_config, parallelism):
        super(BrownianLoadGenerator, self).__init__(load_config, parallelism)

    def loads(self):
        par = self.parallelism
        for i in range(self.periods):
            par = int(par + random.normalvariate(0, 2.0))
            self.memory[i] = par
            yield par
        self.serialise()


