
import numpy as np

class SQLQuery():
    """
    Encapsulates query SELECT COUNT(*) FROM ... WHERE ... 
    Boilerplated from lift/case_studies/{mongodb,mysql}/ 

    Args:
        query_string (str): query string, templated or non-templated
        query_tbl (str): table being queried
        query_cols (list of str): table columns being queried
        index_cols (list of str): currently indexed columns
        is_templated (bool): true if query_string is templated
        sample_fn (callable): samples attributes for query attributes
        tokens (list of str): tokenized query operands + operators 
    
    """

    def __init__(self, query_string, query_tbl=None, query_cols=None, 
                 index_cols = None, is_templated=True, sample_fn=None, tokens=None):

        self.query_string = query_string
        self.query_cols = query_cols
        self.query_tbl = query_tbl  
        self.index_cols = index_cols
        self.is_templated = is_templated
        self.tokens = tokens
        self.sample_fn = sample_fn

    def sample_query(self):
        """
        Samples arguments for a template.

        Returns:
            str, tuple: query, query args
        """
        assert self.is_templated
        return self.query_string, self.sample_fn()

    def full_index_from_query(self):
        """
        Generates input to agent.act for a full index on query attributes.

        Returns:
            dict: dict with index information
        """
        return dict(index=self.query_cols, table=self.query_tbl)

    def as_tokens(self):
        return self.tokens

    def as_csv_row(self):
        """
        Serializes fields for this SQLQuery query

        Returns:
            str: CSV row
        """

        index_cols = '[]' if self.index_cols is None else self.index_cols
        data = [self.query_string, self.query_cols, self.query_tbl, index_cols, self.tokens]
        csv_row = ""
        for i, datum in enumerate(data):
            if i != 0:
                csv_row += ','
            if isinstance(datum, str):
                csv_row += "'" + datum + "'"
            elif isinstance(datum, (list, tuple, np.ndarray)):
                csv_row += "'" + ','.join([str(el) for el in datum]) + "'"
        csv_row += "\n"
        return csv_row
