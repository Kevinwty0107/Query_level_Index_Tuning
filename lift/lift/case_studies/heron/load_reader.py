from queue import Empty

class LoadReader(object):

    def __init__(self, load_config, queue, parallelism):
        self.queue = queue
        self.parallelism = parallelism[load_config['component']]
        self.current_val = self.parallelism
    
    def reset(self):
        self.current_val = self.parallelism

    def read(self):
        while True:
            try:
                msg = self.queue.get(False)
                if msg == 'DONE':
                    break
                self.current_val = msg
                return self.current_val
            except  Empty:
                return self.current_val
        return -1


