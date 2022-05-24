import unittest

import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..'))

from src.common.tpch_util import tpch_tables
from src.common.tpch_workload import TPCHWorkload

from src.dqn.postgres_converter import PostgresConverter
from src.dqn.postgres_schema import PostgresSchema

class TestTPCHWorkload(unittest.TestCase):
    
    experiment_config = {
        "reward_penalty": 0,
        "max_runtime": 1,
        "runtime_weight": 1,
        "max_size": 1,
        "size_weight": 1,
    }
    
    schema_config = {
        "tables": ['customer'],
        "input_sequence_size": 10,
        "max_fields_per_index": 3
    }


    def test_system_to_agent_state(self):
        """
        """

        workload_spec = {
            "tables": ['customer'],
            "n_selections": 3,
            "scale_factor": 3
        }
        workload = TPCHWorkload(spec=workload_spec)
        query = workload.generate_query_template()

        schema = PostgresSchema(schema_config=self.schema_config)
        converter = PostgresConverter(experiment_config=self.experiment_config, schema=schema)

        agent_state = converter.system_to_agent_state(query=query, 
                                                system_context=dict(indices=[['C_ACCTBAL'], ['C_ADDRESS']]))
        print(agent_state.state_value)
        print(query.as_tokens())
        print(schema.system_spec['vocab'])

if __name__ == '__main__':
    unittest.main()