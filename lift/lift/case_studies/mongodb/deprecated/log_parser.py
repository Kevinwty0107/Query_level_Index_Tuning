

class LogParser(object):
    """
    Abstract executable log parser. When executed, the parser reads log messages
    from a source and parses them via its parse method.

    Every case study needs to implement a LogParser.
    """

    def execute(self):
        raise NotImplementedError

    def parse_message(self, message):
        """
        Parses a message and returns a ParsedMessage.

        :param message:
        :return: A ParsedMessage object
        """
        raise NotImplementedError
