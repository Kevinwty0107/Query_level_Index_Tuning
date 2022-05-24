import time
from lift.case_studies.heron.load_process import LoadProcess
class AperiodicLoadProcess(LoadProcess):

    def __init__(self, load_config, parallelism):
        from lift.case_studies.heron import heron_interval_generators
        super(AperiodicLoadProcess, self).__init__(load_config, parallelism)
        self.interval_generator = heron_interval_generators[\
                load_config['load_process_args']['interval_generator']](load_config)

    def run(self):
        for load, interval in zip(self.load_generator.loads(), 
                self.interval_generator.intervals()):
            # sleep for as long as necessary
            time.sleep(interval)
            # just raises the load rather than actually performing the
            # action. This can then be put on the queue asynchronously.
            yield load
        self.finished = True
        self.interval_generator.serialise()

    def replay(self):
        for load, interval in zip(self.load_generator.replay(), 
                self.interval_generator.replay()):
            time.sleep(interval)
            yield load
        
         
