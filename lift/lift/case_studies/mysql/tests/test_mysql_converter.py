import unittest

from lift.case_studies.mysql.mysql_converter import MySQLConverter
from lift.case_studies.mysql.mysql_schema import MySQLSchema
from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload
from lift.case_studies.mysql.tpch_util import tpch_tables, SCALE_FACTOR


class TestMySQLConverter(unittest.TestCase):

    """
    Tests converter.
    """
    config = {
        "max_size": 1,
        "max_runtime": 1,
        "reward_penalty": 0,
        "runtime_weight": 1,
        "size_weight": 1,
        "smooth_runtime": False,
        "reward_regularizer": 0.01

    }
    schema_config = {
        "tables": tpch_tables,
        "input_sequence_length": 10,
        "max_fields_per_index": 3
    }

    def test_state_conversion(self):
        """
        Tests turning a query and a context into a state.

        python -m pytest -s lift/case_studies/mysql/tests/test_mysql_converter.py::TestMySQLConverter::test_state_conversion
        """
        workload_spec = {
            "tables": tpch_tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        query = workload.generate_query_template()
        print(query)
        print("Query columns are =", query.query_columns)

        schema = MySQLSchema(schema_config=self.schema_config)
        converter = MySQLConverter(experiment_config=self.config, schema=schema)

        state = converter.system_to_agent_state(query, system_context=dict(index_columns=[]))
        print(state.state_value)

        # Test with context.
        context = query.full_index_from_query()["index"]
        state = converter.system_to_agent_state(query, system_context=dict(index_columns=context))
        print(state.state_value)

        context = [("L_RETURNFLAG", "ASC")]
        state = converter.system_to_agent_state(query, system_context=dict(index_columns=context))
        print(state.state_value)

        y = query.full_index_from_query()["index"].copy()
        y.extend(context)
        print("context = ", y)
        state = converter.system_to_agent_state(query, system_context=dict(index_columns=y))
        print(state.state_value)

        # 2 single column indices, then a compound indices combining the others.
        # [A, B, [A, B]]
        y = query.full_index_from_query()["index"].copy()
        y.extend([query.full_index_from_query()["index"]])
        print("context = ", y)
        state = converter.system_to_agent_state(query, system_context=dict(index_columns=y))
        print(state.state_value)

    def test_action_to_index(self):
        """
        Tests action mapping agent -> system.
        """
        workload_spec = {
            "tables": tpch_tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)

        schema = MySQLSchema(schema_config=self.schema_config)
        converter = MySQLConverter(experiment_config=self.config, schema=schema)

        # All no-op.
        action = {
            "index_column0": 0,
            "index_column1": 0,
            "index_column2": 0
        }
        data = {
            "query_columns": ["P_PART"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [])

        # One op.
        action = {
            "index_column0": 0,
            "index_column1": 1,
            "index_column2": 0
        }
        data = {
            "query_columns": ["P_PART"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [("P_PART", "ASC")])

        # No self-compounding.
        action = {
            "index_column0": 0,
            "index_column1": 1,
            "index_column2": 1
        }
        data = {
            "query_columns": ["P_PART"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [("P_PART", "ASC")])

        # Batch format
        action = {
            "index_column0": [0],
            "index_column1": [1],
            "index_column2": [1]
        }
        data = {
            "query_columns": ["P_PART"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [("P_PART", "ASC")])

        # Selecting output field 2 if only one input column means no index.
        action = {
            "index_column0": 2,
            "index_column1": 2,
            "index_column2": 2
        }
        data = {
            "query_columns": ["P_PART"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [("P_PART", "DESC")])

        # Combine different fields with no-op in between..
        action = {
            "index_column0": 6,
            "index_column1": 0,
            "index_column2": 2
        }
        data = {
            "query_columns": ["P_PART", "P_BRAND", "P_TYPE"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [("P_TYPE", "DESC"), ("P_PART", "DESC")])

        # In-order.
        action = {
            "index_column0": 1,
            "index_column1": 3,
            "index_column2": 5
        }
        data = {
            "query_columns": ["P_PART", "P_BRAND", "P_TYPE"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [("P_PART", "ASC"), ("P_BRAND", "ASC"), ("P_TYPE", "ASC")])

        # Reverse order (indices are order sensitive).
        action = {
            "index_column0": 5,
            "index_column1": 3,
            "index_column2": 1
        }
        data = {
            "query_columns": ["P_PART", "P_BRAND", "P_TYPE"]
        }

        index = converter.agent_to_system_action(action, data)
        self.assertEqual(index, [("P_TYPE", "ASC"), ("P_BRAND", "ASC"), ("P_PART", "ASC")])

    def test_index_to_action(self):
        """
        Tests mapping: system action -> agent action
        """
        workload_spec = {
            "tables": tpch_tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        query = workload.generate_query_template()

        schema = MySQLSchema(schema_config=self.schema_config)
        converter = MySQLConverter(experiment_config=self.config, schema=schema)

        # Fix query columns.
        query.query_columns = ["P_TYPE", "P_BRAND", "P_PART"]

        # Test no-op.
        index = dict(index=[])
        expected = {
            "index_column0": [0],
            "index_column1": [0],
            "index_column2": [0]
        }
        action = converter.system_to_agent_action(query=query, system_action=index)
        self.assertEqual(action, expected)

        # Test first.
        index = dict(index=[("P_TYPE", "ASC")])
        expected = {
            "index_column0": [1],
            "index_column1": [0],
            "index_column2": [0]
        }
        action = converter.system_to_agent_action(query=query, system_action=index)
        self.assertEqual(action, expected)

        # Test reverse.
        index = dict(index=[("P_PART", "ASC"), ("P_BRAND", "ASC"), ("P_TYPE", "ASC")])
        expected = {
            "index_column0": [5],
            "index_column1": [3],
            "index_column2": [1]
        }
        action = converter.system_to_agent_action(query=query, system_action=index)
        self.assertEqual(action, expected)

        # Test reverse with no-op in between
        index = dict(index=[("P_TYPE", "DESC"), ("P_PART", "DESC")])
        expected = {
            "index_column0": [2],
            "index_column1": [6],
            "index_column2": [0]
        }
        action = converter.system_to_agent_action(query=query, system_action=index)
        self.assertEqual(action, expected)
