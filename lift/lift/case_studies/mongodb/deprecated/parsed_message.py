

class ParsedMessage(object):
    """
    A generic parsed log message returned by the LogParser.
    Consumers of the entry should only use its provided methods
     and not directly read fields.
    """

    def __init__(
        self,
        request,
        runtime=None,
        meta_data=None
    ):
        self.request = request
        self.runtime = runtime
        self.meta_data = meta_data

    def get_query(self):
        """
        Returns raw query

        :return: Raw query dict
        """
        return self.request

    def get_runtime(self):
        """
        Returns runtime of request.

        :return: Runtime in milliseconds
        """
        return self.runtime

    def get_meta_data(self):
        """
        Returns request meta data.

        :return: Meta data dict
        """
        return self.meta_data
