import numpy as np

from lift.case_studies.mysql.tpch_util import tpch_table_columns, column_type_operators, tpch_string_values, \
    random_float, sample_text, tpch_sample_fns, query_delimiter, SCALE_FACTOR
from lift.case_studies.mysql.sql_query import SQLQuery
from lift.case_studies.common.query_workload import QueryWorkload


class TPCHSyntheticWorkload(QueryWorkload):
    """
    We can imagine multiple variants of TPC-H workloads:

    1. Use TPC-H Data and TPC-H queries (do not test generalisation, RL as optimisation)
    2. Use TPC-H data and synthetic randomized templated queries for training and testing
    3. Use TPC-H Data and randomize TPC-H queries by removing random selections from queries.

    This class implements approach (2):

    It also provides randomized demo queries that are meant to be used with a labeling function to generate
    demos.
    """
    def __init__(self, workload_spec):
        """

        Args:
            workload_spec (dict): Spec specifying the complexity of the synthetic queries (e.g. max number of
            selections).
        """
        self.workload_spec = workload_spec
        # Tables to use when generating random queries.
        self.tables = workload_spec.pop("tables")

        # tpch scale factor.
        self.scale_factor = workload_spec.pop("scale_factor")
        self.num_selections = workload_spec.pop("num_selections")
        self.sort_prob = workload_spec.get("sort_prob", 0.5)

    def generate_query_template(self):
        """
        Builds a templated query.

        """
        table = np.random.choice(self.tables)
        query_prefix = "SELECT COUNT(*) FROM {} WHERE".format(table)
        selections = []

        # Column name -> type for this table.
        table_columns = tpch_table_columns[table]
        column_names = list(table_columns.keys())
        token_list = []
        sort_order = []

        num_selections = np.random.randint(1, self.num_selections + 1)
        columns_in_query = np.random.choice(column_names, size=num_selections, replace=False)
        logical_operator = np.random.choice(['AND', 'OR'], size=1)[0]

        base_string = " {} ".format(logical_operator)
        for i in range(num_selections):
            # Select a column from this table
            selection_column = columns_in_query[i]
            descriptor = table_columns[selection_column]
            column_type = descriptor[0]
            # Look up sensible operators for the column type:
            operators_for_type = column_type_operators[column_type]
            selection_operator = np.random.choice(operators_for_type)

            sort = np.random.random() > self.sort_prob
            if sort:
                # Select uniformly a sort order for each column that requires sort.
                # 1 = asc, -1 = desc
                column_sort = "ASC" if np.random.random() > 0.5 else "DESC"
                sort_order.append((selection_column, column_sort))
                token_list.append("{}_{}".format(selection_column, column_sort))
            else:
                token_list.append("{}_ASC".format(selection_column))
            token_list.append(selection_operator)

            # E.g. R_REGIONKEY < %s, arg will be sampled.
            selection = "{} {} '%s'".format(selection_column, selection_operator)
            selections.append(selection)

        selection_string = base_string.join(selections)

        # Combine SELECT COUNTS FROM WHERE selection_1 AND selection _2...
        if len(sort_order) > 0:
            token_list.append("SORT")
            sort_list = []
            for sort_tuple in sort_order:
                sort_list.append("{} {}".format(sort_tuple[0], sort_tuple[1]))
                # Also append to token list.
                # token_list.append("{}_{}".format(sort_tuple[0], sort_tuple[1]))
            sort_string = ", ".join(sort_list)
            query_string = "{} {} ORDER BY {}".format(query_prefix, selection_string, sort_string)
        else:
            query_string = "{} {}".format(query_prefix, selection_string)

        def sample_fn():
            sampled_args = []
            for column in columns_in_query:
                descriptor = table_columns[column]
                column_type = descriptor[0]
                sample_type = descriptor[1]

                # Sample according to tpch value specs.
                sample = None
                if sample_type == "lookup":
                    sample = np.random.choice(tpch_string_values[column])
                elif sample_type == "fixed_range":
                    range_tuple = descriptor[2]
                    if column_type == int:
                        sample = np.random.randint(low=range_tuple[0], high=range_tuple[1])
                    elif column_type == float:
                        sample = random_float(low=range_tuple[0], high=range_tuple[1])
                elif sample_type == "scaled_range":
                    range_tuple = descriptor[2]
                    scaled_low = range_tuple[0] * self.scale_factor
                    scaled_high = range_tuple[1] * self.scale_factor
                    if column_type == int:
                        sample = np.random.randint(low=scaled_low, high=scaled_high)
                    elif column_type == float:
                        sample = random_float(low=scaled_low, high=scaled_high)
                elif sample_type == "text":
                    sample = sample_text()
                elif sample_type == "sample_fn":
                    sample = tpch_sample_fns[column]()
                elif sample_type == "scaled_sample_fn":
                    sample = tpch_sample_fns[column](self.scale_factor)
                else:
                    raise ValueError("No arg sampled for {} with spec {}".format(column, descriptor))

                sampled_args.append(sample)

            # MySQL client wants a tuple.
            return tuple(sampled_args)

        return SQLQuery(query_string, query_table=table,
                        query_columns=columns_in_query, sample_fn=sample_fn, tokens=token_list,
                        sort_order=sort_order, logical_operator=logical_operator)

    @staticmethod
    def query_from_csv(query_csv):
        """
        Converts a serialised representation to a SQLQuery object.

        Args:
            query_csv (list): CSV entry.

        Returns:
            SQLQuery: A SQLQuery object from the serialised representation,
        """
        query_columns = query_csv[1].split(query_delimiter)
        query_table = query_csv[2]
        index_columns = None if query_csv[3] == '[]' else query_csv[3].split(query_delimiter)
        tokens = query_csv[4].split(query_delimiter)
        sorted_columns = None if query_csv[5] == "" else query_csv[5].split(query_delimiter)
        sort_directions = None if query_csv[6] == "" else query_csv[6].split(query_delimiter)
        logical_operator = query_csv[7]

        if sorted_columns:
            sort_order = [(col, order) for col, order in zip(sorted_columns, sort_directions)]
        else:
            sort_order = []

        # Reconstructing query string is is problematic due to escaping arguments -> just rebuild it from columns.
        query_prefix = "SELECT COUNT(*) FROM {} WHERE".format(query_table)

        selections = []
        k = 1
        print("Recovered tokens = ", tokens)
        print("query columns recovered = ", query_columns)
        base_string = " {} ".format(logical_operator)
        for i in range(len(query_columns)):
            # Select a column from this table
            selection_column = query_columns[i]

            # Tokenised version.
            selection_operator = tokens[k]

            # E.g. R_REGIONKEY < %s, arg will be sampled.
            selection = "{} {} '%s'".format(selection_column, selection_operator)
            selections.append(selection)
            k += 2

        selection_string = base_string.join(selections)
        if sort_order:
            sort_list = []
            for sort_tuple in sort_order:
                sort_list.append("{} {}".format(sort_tuple[0], sort_tuple[1]))
                # Also append to token list.
                tokens.append("{}_{}".format(sort_tuple[0], sort_tuple[1]))
            sort_string = ", ".join(sort_list)
            query_string = "{} {} ORDER BY {}".format(query_prefix, selection_string, sort_string)
        else:
            query_string = "{} {}".format(query_prefix, selection_string)

        query = SQLQuery(query_string, query_table=query_table, index_columns=index_columns,
                         query_columns=query_columns, tokens=tokens, sort_order=sort_order)

        # Reconstruct sample fn.
        def sample_fn():
            sampled_args = []
            for column in query.query_columns:
                descriptor = tpch_table_columns[query.query_table][column]
                column_type = descriptor[0]
                sample_type = descriptor[1]

                # Sample according to tpch value specs.
                sample = None
                if sample_type == "lookup":
                    sample = np.random.choice(tpch_string_values[column])
                elif sample_type == "fixed_range":
                    range_tuple = descriptor[2]
                    if column_type == int:
                        sample = np.random.randint(low=range_tuple[0], high=range_tuple[1])
                    elif column_type == float:
                        sample = random_float(low=range_tuple[0], high=range_tuple[1])
                elif sample_type == "scaled_range":
                    range_tuple = descriptor[2]
                    scaled_low = range_tuple[0] * SCALE_FACTOR
                    scaled_high = range_tuple[1] * SCALE_FACTOR
                    if column_type == int:
                        sample = np.random.randint(low=scaled_low, high=scaled_high)
                    elif column_type == float:
                        sample = random_float(low=scaled_low, high=scaled_high)
                elif sample_type == "text":
                    sample = sample_text()
                elif sample_type == "sample_fn":
                    sample = tpch_sample_fns[column]()
                elif sample_type == "scaled_sample_fn":
                    sample = tpch_sample_fns[column](SCALE_FACTOR)
                else:
                    raise ValueError("No arg sampled for {} with spec {}".format(column, descriptor))

                sampled_args.append(sample)

            # MySQL client wants a tuple.
            return tuple(sampled_args)
        query.sample_fn = sample_fn
        return query

    def set_tables(self, tables):
        """
        Updates tables to generate queries from (if need to change between train/test).
        """
        self.tables = tables

    def define_demo_queries(self, num_queries):
        return [self.generate_query_template() for _ in range(num_queries)]

    def define_train_queries(self, num_queries):
        """
        Defines a set of training Queries.

        Returns:
            list: List of SQLQuery objects.
        """
        return [self.generate_query_template() for _ in range(num_queries)]

    def define_test_queries(self, num_queries):
        """
        Defines a set of training Queries.

        Returns:
            list: List of SQLQuery objects.
        """
        return [self.generate_query_template() for _ in range(num_queries)]
