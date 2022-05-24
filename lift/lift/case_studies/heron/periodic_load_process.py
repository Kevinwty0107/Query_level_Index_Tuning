import time
from lift.case_studies.heron.load_process import LoadProcess
class PeriodicLoadProcess(LoadProcess):
    """ 
    Load process which calls the load_generator periodically,
    i.e. at fixed intervals
    """

    def __init__(self, load_config, parallelism):
        super(PeriodicLoadProcess, self).__init__(load_config, parallelism)
        self.period = load_config['load_process_args']['period']
    
    
    def run(self):
        # load_generator is literally a generator
        for load in self.load_generator.loads():
            # wait a fixed time 
            time.sleep(self.period)
            # change the load
            yield load
        self.finished = True
    
    def replay(self):
        for load in self.load_generator.replay():
            time.sleep(self.period)
            yield load
