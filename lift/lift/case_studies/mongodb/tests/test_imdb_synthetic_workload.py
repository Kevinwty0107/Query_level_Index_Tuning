import unittest

from lift.case_studies.mongodb.imdb_synthetic_workload import IMDBSyntheticWorkload
import numpy as np


class TestIMDBSyntheticWorkload(unittest.TestCase):

    """
    Tests if workload generator can generate executable queries.
    """
    def test_sample_fn(self):
        """
        Tests query generation without DB connection.
        """

        workload_spec = {
            "collections": "imdb_all",
            "num_selections": 3
        }
        workload = IMDBSyntheticWorkload(workload_spec=workload_spec)

        template = workload.generate_query_template()
        query_filter_dict = template.query_dict['query_filter']
        key = list(query_filter_dict.keys())[0]

        selections = list(query_filter_dict.values())[0]
        for _ in range(10):
            samples = [selection_fn() for selection_fn in selections]
            sample_dict = {key: samples}
            print(sample_dict)

    def test_workload(self):
        """
        python -m pytest -s test_query_serialisation.py::TestIMDBSyntheticWorkload::test_workload
        """
        from lift.case_studies.mongodb.mongodb_system_environment import MongoDBSystemEnvironment
        exp_config = {
            "database": "imdb",
            "collection": "imdb_all"
        }
        system = MongoDBSystemEnvironment(
            experiment_config=exp_config
        )
        workload_spec = {
            "num_selections": 3
        }
        workload = IMDBSyntheticWorkload(workload_spec=workload_spec)

        num_samples = 25
        repeats = 3
        queries = [workload.generate_query_template() for _ in range(num_samples)]
        # Always use same
        runtimes = []

        print("Executing queries..")
        for query in queries:
            q_time = []
            for _ in range(repeats):
                q_time.append(system.execute(query))
            # print("Executed query {} with runtime {}".format(query, np.mean(q_time)))
            runtimes.append(np.mean(q_time))

        print("Finished executing {} queries (no indexing).".format(num_samples))
        print("Total execution time = {} s ".format(np.sum(runtimes)))
        print("Mean execution time = {} s ".format(np.mean(runtimes)))
        print("Min execution time = {} s".format(np.min(runtimes)))
        print("Max execution time = {} s".format(np.max(runtimes)))
        print("90th percentile execution time = {} s".format(np.percentile(runtimes, 90)))
        print("99th percentile execution time = {} s".format(np.percentile(runtimes, 99)))

