import unittest

import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..'))

from src.common.tpch_util import tpch_tables
from src.common.tpch_workload import TPCHWorkload

class TestTPCHWorkload(unittest.TestCase):
    """
        n.b. from command line:
        python3 -m unittest test_tpch_workload
        python3 -m unittest test_tpch_workload.TestTPCHWorkload
        python3 -m unittest test_tpch_workload.TestTPCHWorkload.test_template_sampling

    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_template_sampling(self):
        """
        TODO: execute each query 
        """

        workload_spec = {
            "tables": tpch_tables,
            "n_selections": 3,
            "scale_factor": 3
        }
        workload = TPCHWorkload(spec=workload_spec)

        for _ in range(10):
            template = workload.generate_query_template()
            query, query_args = template.sample_query()
            print(query % query_args)

if __name__ == '__main__':
    unittest.main()
