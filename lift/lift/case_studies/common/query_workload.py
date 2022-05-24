
from abc import ABC


class QueryWorkload(ABC):
    """
    Defines a Query workload. This could be either programmatically
    or by manually creating a set of query strings (or loading them from an existing benchmark).
    """

    def define_demo_queries(self, *args, **kwargs):
        """
        Depending on the experimental setup, this method can be used to generate
        demonstration queries which are only used for pre-training, but not online.

        Returns:
            list: List of Query objects.
        """
        pass

    def define_train_queries(self, *args, **kwargs):
        """
        Defines a set of training Queries.

        Returns:
            list: List of Query objects.
        """
        pass

    def define_test_queries(self, *args, **kwargs):
        """
        Defines a set of training Queries.

        Returns:
            list: List of Query objects.
        """
        pass

    def generate_query_template(self):
        pass
