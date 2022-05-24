from lift.case_studies.common.query import Query
import time
import numpy as np

from lift.case_studies.mongodb.imdb_util import mongo_query_delimiter, mongo_query_quote_char


class MongoDBQuery(Query):

    def __init__(self, name, query_dict, query_columns, tokens, index_columns=None,
                 selection_operators=None):
        """

        Args:
            name (str): Query id.
            query_dict (dict): Filter dict - query language is in JSON.
            query_columns (list): List of fields in the query.
            tokens (list): Query tokens.
            index_columns (list): List of index tuples.
        """
        self.name = name
        self.query_dict = query_dict
        self.query_columns = query_columns
        self.tokens = tokens
        self.aggregation = self.query_dict["aggregation"]
        self.selection_operators = selection_operators

        # Which fields require sorting?
        self.sort_map = {sort_tuple[0]: sort_tuple[1] for sort_tuple in query_dict["sort_order"]}

        # E.g. [1, -1, ..]
        self.sort_directions = [sort_tuple[1] for sort_tuple in query_dict["sort_order"]]
        self.index_columns = index_columns
        self.sample_fn = self.init_sample_fn()

        self.priority = None

    def init_sample_fn(self):
        """
        This function maps the templated query to an executable function, as opposed to a manually
        construct query function.

        :return: Lambda for execution function
        """
        if self.aggregation == 'limit':
            def query_function(coll):
                query_filter_dict = self.query_dict['query_filter']
                key = list(query_filter_dict.keys())[0]

                selections = list(query_filter_dict.values())[0]
                samples = [selection_fn() for selection_fn in selections]
                sample_dict = {key: samples}

                start = time.monotonic()
                coll.find(sample_dict).limit(10)
                return time.monotonic() - start

        elif self.aggregation == 'count':
            def query_function(coll):
                query_filter_dict = self.query_dict['query_filter']
                key = list(query_filter_dict.keys())[0]

                selections = list(query_filter_dict.values())[0]
                samples = [selection_fn() for selection_fn in selections]
                sample_dict = {key: samples}

                start = time.monotonic()
                coll.find(sample_dict).count()
                return time.monotonic() - start

        elif self.aggregation == 'sort':
            # We always use a limit after sort.
            def query_function(coll):
                query_filter_dict = self.query_dict['query_filter']
                key = list(query_filter_dict.keys())[0]

                selections = list(query_filter_dict.values())[0]
                samples = [selection_fn() for selection_fn in selections]
                sample_dict = {key: samples}

                start = time.monotonic()
                coll.find(sample_dict).sort(self.query_dict['sort_order']).limit(10)
                return time.monotonic() - start
        else:
            raise ValueError("Invalid aggregation {}.".format(self.aggregation))

        return query_function

    def full_index_from_query(self):
        index = []
        for field in self.query_columns:
            if field in self.sort_map:
                # Append 1 or -1 depending on sort map.
                index.append((field, int(self.sort_map[field])))
            else:
                # Otherwise default sort order 1 (ascending):
                index.append((field, 1))
        return dict(index=index)

    def demonstration(self):
        # In case of manually set demos.
        return dict(index=self.index_columns)

    def as_tokens(self):
        return self.tokens

    def as_csv_row(self):
        index_columns = '[]' if self.index_columns is None else self.index_columns
        query_data = [self.query_columns, index_columns, self.tokens, self.name, self.aggregation,
                      self.selection_operators, self.sort_directions]
        csv_row = ""

        for i, value in enumerate(query_data):
            if i != 0:
                csv_row += mongo_query_delimiter
            if isinstance(value, str):
                csv_row += mongo_query_quote_char + value + mongo_query_quote_char
            elif isinstance(value, (list, tuple, np.ndarray)):
                csv_row += mongo_query_quote_char + mongo_query_delimiter.join([str(v) for v in value]) + mongo_query_quote_char

        csv_row += "\n"
        return csv_row

    def explain(self, collection):
        """
        Executes with out aggregation and returns explain.
        """
        query_filter_dict = self.query_dict['query_filter']
        key = list(query_filter_dict.keys())[0]

        selections = list(query_filter_dict.values())[0]
        samples = [selection_fn() for selection_fn in selections]
        sample_dict = {key: samples}
        return collection.find(sample_dict).explain()
