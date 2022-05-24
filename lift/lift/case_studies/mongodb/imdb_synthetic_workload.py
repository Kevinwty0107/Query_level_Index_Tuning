from lift.case_studies.common.query_workload import QueryWorkload
import numpy as np

from lift.case_studies.mongodb.imdb_util import imdb_collection_info, field_type_operators, imdb_sampling_fns, \
    mongo_query_delimiter
from lift.case_studies.mongodb.mongodb_query import MongoDBQuery


class IMDBSyntheticWorkload(QueryWorkload):

    def __init__(self, workload_spec):
        """

        Args:
            workload_spec (dict): Spec specifying the complexity of the synthetic queries (e.g. max number of
            selections).
        """
        self.workload_spec = workload_spec
        # Tables to use when generating random queries.

        self.logical_ops = workload_spec.get("aggregations_operators", ["$or", "$and"])
        self.aggregations = ['sort', 'limit', 'count']
        self.num_selections = workload_spec.pop("num_selections")

        # Probability with which a sort is requested on a tuple.
        self.sort_prob = workload_spec.get("sort_prob", 0.5)
        self.q_index = 0

    def generate_query_template(self):
        """
        Builds a templated query.
        """
        selections = []

        # Column name -> type for this table.
        collection_fields = imdb_collection_info
        column_names = list(collection_fields.keys())

        num_selections = np.random.randint(1, self.num_selections + 1)
        fields_in_query = np.random.choice(column_names, size=num_selections, replace=False)

        aggregation = np.random.choice(self.aggregations)
        logical_op = np.random.choice(self.logical_ops)
        sort_order = []

        # Query is of form { $and: [expr_1, expr_2, ..]}
        # where expr_i is of form: { field_name: { operator: value}}
        token_list = [logical_op]
        selection_operators = []

        def make_func(f):
            return lambda: f

        for i in range(num_selections):
            # Select a column from this table
            selection_column = fields_in_query[i]
            descriptor = collection_fields[selection_column]
            field_type = descriptor[0]

            # Look up sensible operators for the column type:
            operators_for_type = field_type_operators[field_type]
            selection_operator = np.random.choice(operators_for_type)
            selection_operators.append(selection_operator)
            sampling_fn_for_column = imdb_sampling_fns[selection_column]

            if aggregation == "sort":
                sort_dir = 1 if np.random.random() > self.sort_prob else -1
                sort_order.append((selection_column, sort_dir))
                token_list.append("{}_{}".format(selection_column, sort_dir))
            else:
                token_list.append("{}_1".format(selection_column))
            token_list.append(selection_operator)

            # { field_name: { operator: sample() }
            # Sampling function returns fully formed sample as dict.
            def selection_sampling_fn(selection_column=selection_column,selection_operator=selection_operator,
                                      sampling_fn_for_column=sampling_fn_for_column):
                return {selection_column: {selection_operator: sampling_fn_for_column()}}

            # Careful, scope binding of definitions in loop -> without this, all fns
            # would update their values.
            selections.append(make_func(selection_sampling_fn)())

        key = "{}".format(logical_op)
        # {"and": [...]}
        query_filter = {key: selections}

        name = "q_{}".format(self.q_index)
        self.q_index += 1
        query_dict = dict(query_filter=query_filter, aggregation=aggregation, sort_order=sort_order)

        return MongoDBQuery(
            name=name,
            query_columns=fields_in_query,
            query_dict=query_dict,
            tokens=token_list,
            selection_operators=selection_operators
        )

    @staticmethod
    def query_from_csv(query_csv):
        # print("query csv = ", query_csv)
        query_columns = query_csv[0].split(mongo_query_delimiter)
        index_columns = None if query_csv[1] == '[]' else query_csv[1].split(mongo_query_delimiter)
        tokens = query_csv[2].split(mongo_query_delimiter)
        name = query_csv[3]
        aggregation = query_csv[4]
        selection_operators = query_csv[5].split(mongo_query_delimiter)
        assert len(query_columns) == len(selection_operators),\
            "Query columns and selection operators must have same len but" \
                                                               "are columns: {}, operators: {}".format(query_columns, selection_operators)
        sort_directions = query_csv[6].split(mongo_query_delimiter)
        selections = []
        sort_order = []

        def make_func(f):
            return lambda: f

        # Reconstruct sampling fn.
        for i, selection_column in enumerate(query_columns):

            # Operator is in the token list.
            selection_operator = selection_operators[i]
            sampling_fn_for_column = imdb_sampling_fns[selection_column]

            if aggregation == "sort":
                sort_order.append((selection_column, sort_directions[i]))

            # { field_name: { operator: sample() }
            # Sampling function returns fully formed sample as dict.
            def selection_sampling_fn(selection_column=selection_column,selection_operator=selection_operator,
                                      sampling_fn_for_column=sampling_fn_for_column):
                return {selection_column: {selection_operator: sampling_fn_for_column()}}

            # Careful, scope binding of definitions in loop -> without this, all fns
            # would update their values.
            selections.append(make_func(selection_sampling_fn)())

        # First token is logical op
        key = "{}".format(tokens[0])
        query_filter = {key: selections}

        query_dict = dict(
            query_filter=query_filter,
            aggregation=aggregation,
            sort_order=sort_order
        )

        return MongoDBQuery(
            name=name,
            query_columns=query_columns,
            query_dict=query_dict,
            tokens=tokens,
            index_columns=index_columns
        )

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
