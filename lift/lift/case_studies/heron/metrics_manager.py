from metrics_collector import MetricsCollector
import schedule
import multiprocessing as mp
import threading
class MetricsManager:
    """
        Only odd parameter is callback_funcs. This is a list of functions
        which can be used to add each result from a URL query to the 
        overall state being built up for the RL algorithm. Must accept
        as an argument the result of the query function
    """
    def __init__(self, metrics_collector_list, callback_funcs, time):
        self.metric_list = metrics_collector_list
        self.time = time
        self.processes = 4
        self.pool = mp.Pool(processes = self.processes)
        self.callback_funcs = callback_funcs
        self.jobs_scheduled = False
      
    def _query_wrapper(self, query_func, callback_func):
        self.pool.apply_async(query_func, callback=callback_func)

    """
        Must be called once only
    """
    def schedule_jobs(self):
        self.jobs_scheduled = True
        for i in range(len(self.metric_list)):
            schedule.every(self.time).seconds.do(_query_wrapper, 
                    self.metric_list[i].query, self.callback_funcs[i])


    def _run_pending(self):
        if not self.jobs_scheduled:
            raise Exception('Jobs must be scheduled before they can' +\
            'be run for the MetricsManager class'
        schedule.run_pending()

    def run(self):
        self._run_pending()
        t = threading.Timer(self.time, self.run)
        t.daemon = True
        t.start()
