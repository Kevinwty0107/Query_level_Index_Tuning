from multiprocessing import Process, Queue
from lift.case_studies.heron.load_reader import LoadReader
from lift.case_studies.heron.load_process import LoadWriter
class LoadController(object):

    def __init__(self, experiment_config):
        parallelism = experiment_config['parallelism']
        load_config = experiment_config['load_config']
        self.component = load_config['component']
        queue = Queue()
        self.load_reader = LoadReader(load_config, queue, parallelism)
        self.load_writer = LoadWriter(load_config, parallelism, queue)

    def start(self, replay=False):
        if replay:
            self.write_process = Process(target=self.load_writer.replay)
        else:
            self.write_process = Process(target=self.load_writer.run)
        self.write_process.daemon = True
        self.write_process.start()
    
    def reset(self):
        self.load_reader.reset()

    def read(self):
        parallelism = self.load_reader.read()
        return {self.component : parallelism}
