

class MetricsHelper(object):
    """
    Abstract object to handle learning metrics during runs.
    Must be implemented per case study to handle per-system logging and exporting.
    """
    def __init__(self):
        # Instantiate in subclass
        self.metrics = None

    def record_result(self, result):
        """
        Catch all method to record results of any runtime interactions.
        :param result: Dict with keys describing the state_value of the metric and values
            the state_value(s) to record.
        :return:
        """
        if result:
            for key, value in result.items():
                if key in self.metrics:
                    if isinstance(value, list):
                        # Extend list with multiple samples
                        self.metrics[key].extend(value)
                    else:
                        self.metrics[key].append(value)

    def record_observations(self, observations, kwargs):
        """
        Records new observations (states).
        :return:
        """
        raise NotImplementedError

    def export_results(self, path, kwargs):
        """
        Export runtime results.
        """
        raise NotImplementedError

    def serialize_observations(self, states):
        """
        Parse observations to serialized form.
        :param states:
        :return:
        """
        raise NotImplementedError
