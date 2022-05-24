import random as rn

def gen_fixed_sum(total, n):
    terms = rn.sample(range(1, total), n - 1) + [0, total]
    list.sort(terms)
    return [terms[i + 1] - terms[i] for i in range(n)]


class EMA:
    """ 
    Simple class that wraps a metric handler and computes an
    exponential moving average
    """
    def __init__(self, alpha, handler):
        self.alpha = alpha
        self.handler = handler
        self.ema = 0

    def stream(self):
        # query the metric once
        # TODO update this syntax to query correctly
        while True:
            new_value = handler.query()
            # update ema
            self.ema = (1.0 - self.alpha) * self.ema + new_value
            yield self.ema
