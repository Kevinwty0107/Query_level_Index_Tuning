"""
Util methods for parsing messages.
"""
import heapq
import itertools
import yaml
import re

pattern = re.compile('\\[[^]]*]')
SERIAL_TOKEN = '-'


def sort_input(queries):
    # Sort: Longest queries first, then by columns.
    for query in queries:
        cols = query.query_columns
        priority = '{}_{}'.format(len(cols), str(cols))
        query.priority = priority
    return sorted(queries, key=lambda query: query.priority, reverse=True)


def string_to_dict(query_string):
    # processed = query_string.replace("'", "\"")
    try:
        # TODO next time I generate data -> dont insert field names, dont query field names
        if not query_string.__contains__('$or'):
            replaced_string = pattern.sub("replace", query_string)
        else:
            replaced_string = query_string
    except:
        raise ValueError('Error parsing state_value: {}, type = {}'.format(query_string, type(query_string)))

    return yaml.load(replaced_string)


def serialize_ops(op_list):
    return SERIAL_TOKEN.join(op_list)


def deserialize_ops(op_string):
    return op_string.split(SERIAL_TOKEN)


def dict_to_key_list(op_dict, key_list, data=None):
    """
    Takes a dict representing the MongoOP, extracts lists of keys
    targeted by the query.

    :return: List of keys
    """
    keys = []
    for field in token_generator(op_dict, key_list):
        keys.append(field)

    if not keys:
        raise ValueError(
            'Empty key list, op dict was: {}, keylist was: {}, data was: {}'.
            format(op_dict, key_list, data)
        )
    return keys


def query_priority(query_dict, field_names):
    query_columns = list(token_generator(op_dict=query_dict, tokens=field_names))

    # Lead with length - highest length first, then field names
    return '{}_{}'.format(len(query_columns), str(query_dict))


def token_generator(op_dict, tokens):
    """
    Generates all fields from the op dict that are known in the schema.

    :param op_dict:
    :param tokens:
    :return:
    """
    # Top level could be dict or list
    # This needs to be tested
    if isinstance(op_dict, dict):
        # print('Top level dict = {}'.format(op_dict))
        for dict_key, v in sorted(op_dict.items()):
            # print('Dict key before checking field values = {}'.format(dict_key))
            if dict_key in tokens:
                # Key has to be hashable by definition
                # yield the key the query
                # print('Found key ' + str(dict_key))
                yield dict_key
            if isinstance(v, dict):
                # If sub-op is dict, parse
                # print('Sub state_value is dict = {}'.format(v))
                for field in token_generator(v, tokens):
                    yield field
            elif isinstance(v, list):
                # If sub-op is dict, parse
                # print('Sub state_value is list = {}'.format(v))
                for field in token_generator(v, tokens):
                    yield field

    if isinstance(op_dict, list):
        # Have a list that might contain more dicts or lists
        # print('Top level list = {}'.format(op_dict))

        for list_entry in op_dict:
            # print('List entry' + str(list_entry))

            # First check if nested sublist
            if isinstance(list_entry, dict):
                # print('Sub entry is dict = {}'.format(list_entry))

                # If sub-op is dict, parse
                for field in token_generator(list_entry, tokens):
                    yield field
            elif isinstance(list_entry, list):
                # If sub-op is dict, parse
                # print('Sub entry is list = {}'.format(list_entry))
                for field in token_generator(list_entry, tokens):
                    yield field
            elif list_entry in tokens:
                yield list_entry


class PriorityHeap(object):

    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def add_task(self, task, priority=0):
        # Add a new task or update the priority of an existing task
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        # Mark an existing task as REMOVED.  Raise KeyError if not found.
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        # Remove and return the lowest priority task. Raise KeyError if empty.
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
