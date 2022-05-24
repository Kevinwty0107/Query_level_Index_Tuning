import unittest

from lift.case_studies.mongodb.imdb_synthetic_workload import IMDBSyntheticWorkload
from lift.case_studies.mongodb.mongo_converter import MongoConverter
from lift.case_studies.mongodb.mongo_schema import MongoSchema


class TestMongoDBLConverter(unittest.TestCase):

    """
    Tests converter.
    """
    config = {
        "max_size": 1,
        "max_runtime": 1,
        "reward_penalty": 0,
        "runtime_weight": 1,
        "size_weight": 1,
        "state_mode": "index_net",
        "runtime_regularizer": 0.01,
        "size_regularizer": 0.01,

    }
    schema_spec = {
        "input_sequence_length": 20,
        "max_fields_per_index": 3,
        "collections": "imdb_all"
    }
    schema_config = {
        "titleId": ["string", 2, 30],
        "ordering": ["int", 0, 1000000],
        "title": ["string", 2, 30],
        "region": ["string", 2, 30],
        "language": ["string", 2, 30],
        "types": ["string_array", 25],
        "attributes": ["string_array", 25],
        "isOriginalTitle": ["bool"],
        "tconst": ["string", 2, 30],
        "titleType": ["string", 2, 30],
        "primaryTitle": ["string", 2, 30],
        "originalTitle": ["string", 2, 30],
        "isAdult": ["bool"],
        "startYear": ["date", 10],
        "endYear": ["date", 10],
        "runtimeMinutes": ["int", 0, 1000],
        "genres": ["string_array", 25],
        "averageRating": ["int", 0, 1000000],
        "numVotes": ["int", 0, 1000000]
    }

    def test_state_conversion(self):
        """
        Tests turning a query and a context into a state.

        python -m pytest -s lift/case_studies/mysql/tests/test_mysql_converter.py::TestMongoDBLConverter::test_state_conversion
        """
        workload = IMDBSyntheticWorkload(workload_spec=dict(num_selections=3))
        queries = workload.define_train_queries(5)

        schema = MongoSchema(schema_config=self.schema_config, schema_spec=self.schema_spec, mode='index_net')
        converter = MongoConverter(experiment_config=self.config, schema=schema)

        for query in queries:
            print(query.query_columns)
            # No context.
            state = converter.system_to_agent_state(query, system_context=dict(index_columns=[]))
            print(state.state_value)

            # Simple context.
            context = [query.full_index_from_query()['index']]
            state = converter.system_to_agent_state(query, system_context=dict(index_columns=context))
            print(state.state_value)

            # Add extra unrelated token.
            context = [query.full_index_from_query()['index']] + [("numVotes", 1)]
            state = converter.system_to_agent_state(query, system_context=dict(index_columns=context))
            print(state.state_value)

            # Context is not wrapped as list, so [(field, 1), (field, 2)] are treated as separate indices,
            # not one compound index.
            context = query.full_index_from_query()['index']
            state = converter.system_to_agent_state(query, system_context=dict(index_columns=context))
            print(state.state_value)
