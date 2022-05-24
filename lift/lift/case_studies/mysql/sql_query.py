import numpy as np

from lift.case_studies.common.query import Query
from lift.case_studies.mysql.tpch_util import query_quote_char, query_delimiter


class SQLQuery(Query):

    def __init__(self, query_string, query_columns=None, query_table=None,
                 index_columns=None, is_templated=True, sample_fn=None, tokens=None, sort_order=None,
                 logical_operator=None):
        """
        Generic SQL query and index data for demonstrations.

        Args:
            query_string (str): Query string.
            query_columns (list): List of query columns.
            index_columns (list): List of indexed columns.
            query_table (str): Table this query selects from.
            is_templated: If true, query string has placeholders for args.
            sample_fn (callable): Sample function that provides samples for query args.
            tokens (list): List of tokens in this query. A token is a string for a column or operator in the query.
            sort_order (list):  List of sorted directions.
        """
        self.query_string = query_string
        self.query_columns = list(query_columns)
        self.query_table = query_table
        self.index_columns = index_columns
        self.is_templated = is_templated
        self.tokens = tokens
        self.sample_fn = sample_fn
        self.logical_operator = logical_operator
        # ASC, DESC, ..
        self.sort_map = {sort_tuple[0]: sort_tuple[1] for sort_tuple in sort_order}
        self.sorted_columns = [sort_tuple[0] for sort_tuple in sort_order]
        self.sort_order = [sort_tuple[1] for sort_tuple in sort_order]

        self.priority = None

    def sample_query(self):
        """
        Samples arguments for query.

        Returns:
            str, tuple: Query string and tuple of arguments.
        """
        assert self.is_templated
        return self.query_string, self.sample_fn()

    def full_index_from_query(self):
        """
        Generates the input to the act method to create a full index
        for all columns in the query.

        Returns:
            dict: Index dict.
        """
        index = []
        for field in self.query_columns:
            if field in self.sort_map:
                # Append ASC or DESC depending on sort map.
                index.append((field, self.sort_map[field]))
            else:
                # Otherwise default sort order 1 (ascending):
                index.append((field, "ASC"))
        return dict(
            index=index,
            table=self.query_table
        )

    def as_tokens(self):
        return self.tokens

    def as_csv_row(self):
        """
        Serialisable representation of this query.

        Returns:
            str: CSV row of fields.
        """
        # All fields -> sample fn has to be reconstructed.
        index_columns = '[]' if self.index_columns is None else self.index_columns
        query_data = [self.query_string, self.query_columns, self.query_table, index_columns, self.tokens,
                      self.sorted_columns, self.sort_order, self.logical_operator]
        csv_row = ""

        for i, value in enumerate(query_data):
            if i != 0:
                csv_row += query_delimiter
            if isinstance(value, str):
                csv_row += query_quote_char + value + query_quote_char
            elif isinstance(value, (list, tuple, np.ndarray)):
                csv_row += query_quote_char + query_delimiter.join([str(v) for v in value]) + query_quote_char

        csv_row += "\n"
        return csv_row

    def __repr__(self):
        return self.query_string
