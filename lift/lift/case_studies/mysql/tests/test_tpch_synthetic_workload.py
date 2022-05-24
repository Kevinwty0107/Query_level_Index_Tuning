import unittest

from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload
from lift.case_studies.mysql.tpch_util import tpch_tables, tpch_sample_fns, SCALE_FACTOR


class TestTPCHSyntheticWorkload(unittest.TestCase):

    """
    Tests if workload generator can generate executable queries.
    """
    def test_sample_fn(self):
        """
        Tests query generation without DB connection.
        """

        workload_spec = {
            "tables": tpch_tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)

        for _ in range(100):
            template = workload.generate_query_template()
            print(template.sample_query())

    def test_phone_numbers(self):
        """
        Tests correct query string format when querying phone numbers.

        python -m pytest -s lift/case_studies/mysql/tests/test_tpch_synthetic_workload.py::TestTPCHSyntheticWorkload::test_phone_numbers
        """

        # Test phone formatted with other args.
        from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        query = """SELECT COUNT(*) FROM CUSTOMER WHERE C_ADDRESS = '%s' AND C_PHONE = '%s' AND C_ACCTBAL = '%s';"""
        query_args = tuple([tpch_sample_fns["C_ADDRESS"](), tpch_sample_fns["C_PHONE"](),
                            tpch_sample_fns["C_ACCTBAL"]()])
        runtime = system.execute(query, query_args)
        print("Executed query = {} with args {}, runtime: {} s".format(query, query_args, runtime))

    def test_dates(self):
        """
        Tests correct query string format when querying dates.
        """

        #
        pass

    def test_failed_query(self):
        # Remove after debugging

        # python -m pytest -s lift/case_studies/mysql/tests/test_tpch_synthetic_workload.py::TestTPCHSyntheticWorkload::test_failed_query
        from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        query = """SELECT COUNT(*) FROM LINEITEM WHERE L_LINENUMBER = '%s' AND L_COMMENT = '%s' AND L_PARTKEY > '%s';"""
        query_args = (7, 'silent', 574649)
        runtime = system.execute(query, query_args)
        print("Executed query = {} with args {}, runtime: {} s".format(query, query_args, runtime))

    def test_execute_sample_fn(self):
        """
        Generate queries, execute them, measure runtime.
        python -m pytest -s lift/case_studies/mysql/tests/test_tpch_synthetic_workload.py::TestTPCHSyntheticWorkload::test_execute_sample_fn
        """
        # Imported here so can run other test without mysql install.
        from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        workload_spec = {
            "tables": tpch_tables,
            "scale_factor": 1,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        for _ in range(100):
            template = workload.generate_query_template()
            query, query_args = template.sample_query()
            runtime = system.execute(query, query_args)
            print("Executed query = {} with args {}, runtime: {} s".format(query, query_args, runtime))
