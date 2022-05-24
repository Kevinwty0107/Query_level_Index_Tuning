from lift.pretraining.demonstration_rule import DemonstrationRule

"""Demonstrations rules for MySQL."""


class MySQLPositiveRule(DemonstrationRule):

    def generate_demonstration(self, states, context=None):
        """
        Create full index (does not include context).

        Args:
            states (SQLQuery): SQLQuery to use.
            context (Optional[list]): Index context.
        """
        return states.full_index_from_query()


class MySQLNoop(DemonstrationRule):

    def generate_demonstration(self, states, context=None):
        """
        Defaults  to no-op to discourage unneeded  indices.

        Args:
            states (SQLQuery): SQLQuery to use.
            context (Optional[list]): Index context.
        """
        return dict(index=[])

    def __repr__(self):
        return "No-op rule"


class MySQLSingleColumn(DemonstrationRule):

    def generate_demonstration(self, states, context=None):
        """
        Creates the first missing single-column index.

        Args:
            states (SQLQuery): SQLQuery to use.
            context (Optional[list]): Index context.
        """
        columns = states.query_columns
        # Is there a missing index tuple? Just create that.
        for c in columns:
            if (c, "ASC") not in context:
                return dict(index=[(c, "ASC")])
        # Otherwise no-op.
        return dict(index=[])

    def __repr__(self):
        return "Single column"


class MySQLPrefixRule(DemonstrationRule):

    def generate_demonstration(self, states, context=None):
        """
        Creates full index if no prefixing possible.

        Prefixing is checked by seeing if the full index for the query
        would be already available via prefixing.

        The obvious deficit is that the full index may not be optimal.

        Args:
            states (MongoDBQuery): MongoDBQuery to use.
            context (Optional[list]): Index context.
        """
        # What would the full index be for this query?
        full_index = states.full_index_from_query()['index']
        inverted_full_index = []
        for v in full_index:
            if v[1] == "ASC":
                inverted_full_index.append((v[0], "DESC"))
            elif v[1] == "DESC":
                inverted_full_index.append((v[0], "ASC"))
            else:
                raise ValueError("Illegal sort.")

        # Is the full index contained in an existing index?
        match = False
        for index_tuple in context:
            # A full index with the same columns already exists.
            if full_index == index_tuple:
                match = True
                break
            # The inverted full index exists.
            if inverted_full_index == index_tuple:
                match = True
                break
            # The full index for this query is a prefix of an existing index.
            # The inverted full index for this query is a prefix.
            full_str = '_'.join([str(v) for v in full_index])
            inverted_full_str = '_'.join([str(v) for v in inverted_full_index])
            existing_str = '_'.join([str(v) for v in index_tuple])
            if existing_str.startswith(full_str):
                match = True
                break
            if existing_str.startswith(inverted_full_str):
                match = True
                break

        if match:
            return dict(index=[])
        else:
            return states.full_index_from_query()

    def __repr__(self):
        return "Prefix rule."


class MySQLNegativeRule(DemonstrationRule):
    """This rule discourages the use of wrong prefixing, i.e. it show not to use
      any reverse order indices."""

    def generate_demonstration(self, states, context=None):
        return dict(index=list(reversed(states.full_index_from_query()["index"])))

    def __repr__(self):
        return "Negative rule: Avoid reversing for prefixing."
