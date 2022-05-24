from abc import ABC


class Query(ABC):
    """
    Abstract query object defining API for query case studies (for workflow compatibility across
    case studies).
    """

    def sample_query(self):
        """
        Sample parameters (or return fixed parameters if query not templated).
        Returns:
            any: Templated sample for execution.
        """
        raise NotImplementedError

    def full_index_from_query(self):
        """
        Helper to describe the full index for this query.
        Returns:
            Union(dict, list): Index representation.
        """
        raise NotImplementedError

    def as_tokens(self):
        """
        Provides a tokenised version of this query to avoid parsining.

        Returns:
            list: Token list.
        """
        raise NotImplementedError

    def as_csv_row(self):
        """
        Returns serialisable string.

        Returns:
            str: CSV string.
        """
        raise NotImplementedError
