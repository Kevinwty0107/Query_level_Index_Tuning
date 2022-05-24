import unittest
import numpy as np
import time
from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload
from lift.case_studies.mysql.tpch_util import tpch_tables, tpch_sample_fns, SCALE_FACTOR


class TestTPCHSyntheticBenchmark(unittest.TestCase):

    """
     This test class is meant to evaluate performance characteristics of a specific workload.

     Use to calibrate and identify baseline settings:

     - For a given scale factor and database size:
     - Evaluate how long it takes to run n queries
     - Evaluate how long it takes to create k in [0, n] indices.
     - Evaluate how much indexing improves runtime to find database size/query complexity where indexing
      makes a noticeable difference.
    """

    def test_workload(self):
        """
        Generate queries, execute them, measure runtime.
        python -m pytest -s lift/case_studies/mysql/tests/test_tpch_synthetic_benchmark.py::TestTPCHSyntheticBenchmark::test_workload
        """
        # Imported here so can run other test without mysql install.
        from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        workload_spec = {
            "tables": ["ORDERS", "LINEITEM", "PART"],
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        num_samples = 25
        repeats = 3
        queries = [workload.generate_query_template() for _ in range(num_samples)]
        # Always use same
        samples = [query.sample_query() for query in queries]
        runtimes = []

        print("Executing queries..")
        for sample in samples:
            query, query_args = sample
            q_time = []
            for _ in range(repeats):
                q_time.append(system.execute(query, query_args))
            # print("Executed query {} with runtime {}".format(query, np.mean(q_time)))
            runtimes.append(np.mean(q_time))

        print("Finished executing {} queries (no indexing).".format(num_samples))
        print("Total execution time = {} s ".format(np.sum(runtimes)))
        print("Mean execution time = {} s ".format(np.mean(runtimes)))
        print("Min execution time = {} s".format(np.min(runtimes)))
        print("Max execution time = {} s".format(np.max(runtimes)))
        print("90th percentile execution time = {} s".format(np.percentile(runtimes, 90)))
        print("99th percentile execution time = {} s".format(np.percentile(runtimes, 99)))

        # Now create a full index for every query, rerun.
        print("Creating indices..")
        start = time.perf_counter()
        for query in queries:
            index = query.full_index_from_query()
            system.drop_index(index)
            # print("Creating index {} for query {}.".format(index, query))
            system.act(index)
        end = time.perf_counter() - start
        print("Time to create {} indices: {} s.".format(len(queries), end))
        print("Mean time to create index: {} s".format(end / float(len(queries))))

        runtimes_full_indexing = []
        print("Executing queries..")
        for sample in samples:
            query, query_args = sample
            q_time = []
            for _ in range(repeats):
                q_time.append(system.execute(query, query_args))
            # print("Executed query {} with runtime {}".format(query, np.mean(q_time)))
            runtimes_full_indexing.append(np.mean(q_time))

        print("Finished executing {} queries (full indexing).".format(num_samples))
        print("Total execution time = {} s ".format(np.sum(runtimes_full_indexing)))
        print("Mean execution time = {} s ".format(np.mean(runtimes_full_indexing)))
        print("Min execution time = {} s".format(np.min(runtimes_full_indexing)))
        print("Max execution time = {} s".format(np.max(runtimes_full_indexing)))
        print("90th percentile execution time = {} s".format(np.percentile(runtimes_full_indexing, 90)))
        print("99th percentile execution time = {} s".format(np.percentile(runtimes_full_indexing, 99)))

        # Identify most expensive query -> index not used potentially -> check in CLI via explain
        i = int(np.argmax(runtimes_full_indexing))
        print("Most expensive query is (runtime no index = {} s, runtime full index = {} s):".format(
            runtimes[i], runtimes_full_indexing[i]))
        most_expensive = samples[i]
        query, query_args = most_expensive
        print(query % query_args)
        print("Explaining query = ")
        print(system.explain_query(query, query_args))

        for query in queries:
            index = query.full_index_from_query()
            system.drop_index(index)

    def test_query_variance(self):
        """
        Tests performance variance of a single query to understand impact of sample values.

        python -m pytest -s lift/case_studies/mysql/tests/test_tpch_synthetic_benchmark.py::TestTPCHSyntheticBenchmark::test_query_variance
        """
        from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        workload_spec = {
            "tables": ["LINEITEM"],
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }

        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        num_samples = 100
        query_template = workload.generate_query_template()

        runtimes = []
        q_time = []
        for _ in range(num_samples):
            query, query_args = query_template.sample_query()
            q_time.append(system.execute(query, query_args))
        # print("Executed query {} with runtime {}".format(query, np.mean(q_time)))
        runtimes.append(np.mean(q_time))

        query, query_args = query_template.sample_query()
        print("Finished executing query = ", str(query_template))
        print("Mean execution time = {} s ".format(np.mean(runtimes)))
        print("Min execution time = {} s".format(np.min(runtimes)))
        print("Max execution time = {} s".format(np.max(runtimes)))
        print("Explaining query example = ")
        print(system.explain_query(query, query_args))