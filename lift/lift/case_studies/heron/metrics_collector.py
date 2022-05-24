import logging 
import threading
import requests
"""
Simple class which can query a particular web address with some
argument dictionary and run a callback on the result.
"""
class MetricsCollector:
    
    def __init__(self, urls, arguments, parsing_fn, agg_fn = lambda x: x,
            result_fn = lambda x: x, verbose = False, print_json = False, 
            max_it = 5,**kwargs):
        # interval between queries of the metric
        self.logger = logging.getLogger(__name__)
        self.urls = urls
        self.parsing_fn = parsing_fn
        self.arguments = arguments
        self.agg_fn = agg_fn
        self.result_fn = result_fn
        self.kwargs = kwargs
        self.verbose = verbose
        self.print_json = print_json
        self.max_it = max_it
    
    def query(self, urls=None, arguments = None):
        if arguments is not None:
            self.arguments = arguments
        if urls is not None:
            self.urls = urls
        results = []
        for i in range(len(self.urls)):
            self.logger.debug(self.urls[i])
            self.logger.debug(self.arguments[i])
            r = requests.get(self.urls[i], params=self.arguments[i])
            self.logger.debug('JSON: {}'.format(r.json()))
            if 'result' in r.json():
                it = 0
                returned = None
                while not returned and it < self.max_it:
                    returned = self.parsing_fn(r.json()['result'], self.agg_fn, 
                        **self.kwargs)
                    it = it + 1
                results.append(returned)
            else:
                self.logger.error('{}'.format(r.json()['message']))
                return None
        self.logger.info('Results: {}'.format(results))
        return self.result_fn(results)
       

