
class LoadProcess(object):

    def __init__(self, load_config, parallelism):
        self.finished = False
        from lift.case_studies.heron import heron_load_generators

        self.load_generator = heron_load_generators[\
                load_config['load_generator']](load_config, parallelism)

    def run(self):
        raise NotImplementedError
    
    def replay(self):
        """
        Replays the previous loads etc. in order to allow direct comparison
        of methods
        """
        raise NotImplementedError
    def finished(self):
        return self.finished

class LoadWriter(object):

    def __init__(self, load_config, parallelism, queue):
        self.queue = queue
        from lift.case_studies.heron import heron_load_processes
        self.parallelism = parallelism[load_config['component']]
        self.load_process = heron_load_processes[load_config['load_process']]\
                (load_config, parallelism)

    def run(self):
        #self.queue.put(self.parallelism)
        for load in self.load_process.run():
            self.queue.put(load)
        self.queue.put('DONE')

    def replay(self):
        #self.queue.put(self.parallelism)
        for load in self.load_process.replay():
            self.queue.put(load)
        self.queue.put('DONE')

