import random
import numpy as np
class IntervalGenerator(object):

    def __init__(self, load_config):
        self.total_calls = load_config['load_generator_args']['periods']
        self.memory = np.zeros((self.total_calls,))
        self.file_name = load_config['interval_generator_args']['save_file'] 

    def intervals(self):
        raise NotImplementedError

    def replay(self):
        """
        Replays the previous timings.
        """
        # necessary because if this is launched in a new process then it
        # the memory will be all zeros.
        self.restore()
        for i in range(self.total_calls):
            yield self.memory[i]
    
    def serialise(self):
        """
        Save the generated intervals to file for restoration later.
        MUST be called at the end of the intervals() method because in
        the case this is called in a new process, the memory will be
        wiped
        """
        np.save(self.file_name, self.memory)

    def restore(self):
        """
        Restore the memory from the file
        """
        self.memory = np.load(self.file_name)

class ConstantIntervalGenerator(IntervalGenerator):

    def __init__(self, load_config):
        super(ConstantIntervalGenerator, self).__init__(load_config)
        self.period = load_config['interval_generator_args']['period']

    def intervals(self):
        for i in range(self.total_calls):
            self.memory[i] = self.period
            yield self.period

class UniformIntervalGenerator(IntervalGenerator):

    def __init__(self, load_config):
        super(UniformIntervalGenerator, self).__init__(load_config)
        self.start_range = load_config['interval_generator_args']['start_range']
        self.end_range = load_config['interval_generator_args']['end_range']

    def intervals(self):
        for i in range(self.total_calls):
            r = random.randint(start_range, end_range)
            self.memory[i] = r
            yield r

