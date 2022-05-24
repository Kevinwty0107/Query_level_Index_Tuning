try:
    from abc import ABC
except ImportError:
    import sys
    if sys.version_info[0] == 2: # opentuner runs 2 not 3
        ABC = object

class SQLWorkload(ABC):
    """
    Abstract Base Class that encapsulates the SQL workload.
    """

    def define_demo_queries(self, *args, **kwargs):
        """
        """
        pass

    def define_train_queries(self, *args, **kwargs):
        """
        """
        pass

    def define_test_queries(self, *args, **kwargs):
        """
        """
        pass
