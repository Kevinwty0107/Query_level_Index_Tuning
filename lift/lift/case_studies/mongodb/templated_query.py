import time


class TemplatedQuery(object):

    def __init__(
            self,
            collection=None,
            query_dict=None,
            index_tuples=None,
            log_context=None,
            sample_values=False,
            query_helper=None
    ):
        self.collection = collection
        self.query_dict = query_dict
        self.index_tuples = index_tuples
        self.log_context = log_context
        self.sample_values = sample_values
        if self.sample_values:
            self.query_helper = query_helper
        self.sample_fn = self.init_sample_fn()
        self.priority = -1

    def execute(self):
        return self.sample_fn(self.collection)

    def get_query_dict(self):
        """
        Returns serialized version
        :return:
        """
        assert self.query_dict is not None
        return self.query_dict

    def get_index_tuples(self):
        """
        Returns correct index for a query given its context (or no context.)
        """
        assert self.index_tuples is not None
        return self.index_tuples

    def get_context(self):
        """
        Returns query context, e.g. existing indices. This is serialised to the
        context can be input as part of the state.
        """
        return self.log_context

    def init_sample_fn(self):
        """
        This function maps the templated query to an executable function, as opposed to a manually
        construct query function.

        :return: Lambda for execution function
        """
        aggregation = self.query_dict['aggregation']

        if aggregation == 'limit':
            self.full_query_string = 'coll.find({}).limit(10)'.format(self.query_dict['query_filter'])

            def query_function(coll):
                if self.sample_values:
                    query_filter_dict = self.replace_dict(self.query_dict['query_filter'])
                else:
                    query_filter_dict = self.query_dict['query_filter']
                start = time.monotonic()
                coll.find(query_filter_dict).limit(10)
                return time.monotonic() - start

        elif aggregation == 'count':
            self.full_query_string = 'coll.find({}).count()'.format(self.query_dict['query_filter'])

            def query_function(coll):
                if self.sample_values:
                    query_filter_dict = self.replace_dict(self.query_dict['query_filter'])
                else:
                    query_filter_dict = self.query_dict['query_filter']
                start = time.monotonic()
                coll.find(query_filter_dict).count()
                return time.monotonic() - start

        elif aggregation == 'sort':
            self.full_query_string = 'coll.find({}).sort({}).limit(10)'.format(
                self.query_dict['query_filter'], self.query_dict['sort_order'])

            # We always use a limit after sort.
            def query_function(coll):
                if self.sample_values:
                    query_filter_dict = self.replace_dict(self.query_dict['query_filter'])
                else:
                    query_filter_dict = self.query_dict['query_filter']
                start = time.monotonic()
                coll.find(query_filter_dict).sort(self.query_dict['sort_order']).limit(10)
                return time.monotonic() - start
        else:
            raise ValueError("Invalid aggregation {}.".format(aggregation))

        return query_function

    def replace_dict(self, filter_dict):
        result_dict = dict()
        self.recursive_replace(filter_dict, result_dict)
        return result_dict

    def recursive_replace(self, filter_dict, result_dict):
        """
        Replaces values in subexpressions with calls to sample function.

        :param filter_dict: Original dict
        :param result_dict: Sampling dict
        :return:
        """
        if isinstance(filter_dict, dict):
            for key, value in filter_dict.items():
                # Should have arrived at subexpression of form
                #  {attribute: {operator: state_value}}
                if self.query_helper.schema_contains(key):
                    # Make sure this is dict.
                    assert isinstance(value, dict)
                    # Should only be one
                    for operator in value:
                        # Replace state_value with sample function.
                        result_dict[key] = {operator: self.query_helper.sample_value(key)}
                else:
                    # If state_value is a a list, we want to preserve this:
                    if isinstance(value, list):
                        result_list = []
                        result_dict[key] = result_list
                        for list_value in value:
                            result_list.append(self.recursive_replace(list_value, dict()))
                    elif isinstance(value, dict):
                        # Single state_value, recursively replace
                        result_dict[key] = self.recursive_replace(value, dict())
                return result_dict
        elif isinstance(filter_dict, list):
            for list_entry in filter_dict:
                return self.recursive_replace(list_entry, dict())

    def __repr__(self):
        return self.full_query_string
