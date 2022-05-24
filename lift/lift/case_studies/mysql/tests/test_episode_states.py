import unittest

from lift.case_studies.mysql.mysql_converter import MySQLConverter
from lift.case_studies.mysql.mysql_schema import MySQLSchema
from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload
from lift.case_studies.mysql.tpch_util import tpch_tables, SCALE_FACTOR


class TestEpisodeStates(unittest.TestCase):

    """
    Tests converter.
    """
    config = {
        "max_size": 1,
        "max_runtime": 1,
        "reward_penalty": 0,
        "runtime_weight": 1,
        "size_weight": 1,
        "smooth_runtime": False

    }
    schema_config = {
        "tables": tpch_tables,
        "input_sequence_length": 10,
        "max_fields_per_index": 3
    }

    def test_states(self):
        """
        Tests turning a query and a context into a state.

        python -m pytest -s lift/case_studies/mysql/tests/test_mysql_converter.py::TestMySQLConverter::test_state_conversion
        """
        workload_spec = {
            "tables": ["LINEITEM"],
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        queries= [workload.generate_query_template() for _ in range(20)]

        schema = MySQLSchema(schema_config=self.schema_config)
        converter = MySQLConverter(experiment_config=self.config, schema=schema)

        context = []
        for query in queries:
            print("Raw query = ", query.query_string)
            state = converter.system_to_agent_state(
                query=query, system_context=dict(index_columns=context)
            )
            context.append(query.full_index_from_query()["index"])
