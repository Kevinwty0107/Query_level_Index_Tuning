import unittest
import csv
from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload
from lift.case_studies.mysql.tpch_util import tpch_tables, SCALE_FACTOR, query_delimiter, query_quote_char


class TestQuerySerialisation(unittest.TestCase):
    """
    Tests export and reconstruction of templated query objects.
    """
    # Local test dir for files.
    tmp_folder = "tmp/"

    def test_csv_io(self):
        """
        Tests csv import and export (no db connection needed).

        python -m pytest -s test_query_serialisation.py::TestQuerySerialisation::test_csv_io
        """
        workload_spec = {
            "tables": tpch_tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        query_template = workload.generate_query_template()

        csv_query = query_template.as_csv_row()

        print("CSV conversion:")
        print("Query is = ", query_template)
        print("CSV serialisation is = ", csv_query)

        path = self.tmp_folder + 'serialisation_test.csv'
        with open(path, 'w', newline='') as f:
            f.write(csv_query)

            reconstructed_query = None
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=query_delimiter, quotechar=query_quote_char)
            for row_data in reader:
                print("Retrieved from csv = ", row_data)
                reconstructed_query = TPCHSyntheticWorkload.query_from_csv(row_data)
                print("Reconstructed query = ", str(reconstructed_query))

        self.assertEqual(query_template.query_string, reconstructed_query.query_string)
        print("Sample from original query = ")
        print(query_template.sample_query())
        print("Sample from reconstructed query = ")
        print(reconstructed_query.sample_query())

    def test_executing_reconstructed(self):
        """
        Tests if a reconstructed query can be executed correctly.

        python -m pytest -s test_query_serialisation.py::TestQuerySerialisation::test_executing_reconstructed
        """
        from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
        system = MySQLSystemEnvironment(user="mks40", db="tpch", all_tables=tpch_tables)

        workload_spec = {
            "tables": tpch_tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": 3
        }
        workload = TPCHSyntheticWorkload(workload_spec=workload_spec)
        query_template = workload.generate_query_template()

        csv_query = query_template.as_csv_row()
        path = self.tmp_folder + 'serialisation_test.csv'
        with open(path, 'w', newline='') as f:
            f.write(csv_query)

        reconstructed_query = None
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=query_delimiter, quotechar=query_quote_char)
            for row_data in reader:
                reconstructed_query = TPCHSyntheticWorkload.query_from_csv(row_data)

        # Now try executing original and reconstruction
        query_string, args = query_template.sample_query()
        runtime = system.execute(query_string, args)
        # Runtimes are not expected to be the same.
        print("Original query run time = ", runtime)

        reconstructed_query_string, args = reconstructed_query.sample_query()
        self.assertEqual(query_string, reconstructed_query_string)
        runtime = system.execute(reconstructed_query_string, args)
        print("Reconstructed query run time = ", runtime)

