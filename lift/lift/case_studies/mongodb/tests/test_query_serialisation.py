import unittest
import csv

from lift.case_studies.mongodb.imdb_synthetic_workload import IMDBSyntheticWorkload
from lift.case_studies.mongodb.imdb_util import mongo_query_delimiter, mongo_query_quote_char
from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload


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
            "num_selections": 3
        }
        workload = IMDBSyntheticWorkload(workload_spec=workload_spec)
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
            reader = csv.reader(csvfile, delimiter=mongo_query_delimiter, quotechar=mongo_query_quote_char)
            for row_data in reader:
                print("Retrieved from csv = ", row_data)
                reconstructed_query = IMDBSyntheticWorkload.query_from_csv(row_data)
                print("Reconstructed query = ", str(reconstructed_query))

        print("Sample from original query = ")
        for _ in range(3):
            query_filter_dict = query_template.query_dict['query_filter']
            key = list(query_filter_dict.keys())[0]

            selections = list(query_filter_dict.values())[0]
            samples = [selection_fn() for selection_fn in selections]
            sample_dict = {key: samples}
            print(sample_dict)

        print("Sample from reconstructed query = ")
        for _ in range(3):
            query_filter_dict = reconstructed_query.query_dict['query_filter']
            key = list(query_filter_dict.keys())[0]

            selections = list(query_filter_dict.values())[0]
            samples = [selection_fn() for selection_fn in selections]
            sample_dict = {key: samples}
            print(sample_dict)

    def test_executing_reconstructed(self):
        """
        Tests if a reconstructed query can be executed correctly.

        python -m pytest -s test_query_serialisation.py::TestQuerySerialisation::test_executing_reconstructed
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
        query_template = workload.generate_query_template()

        csv_query = query_template.as_csv_row()
        path = self.tmp_folder + 'serialisation_test.csv'
        with open(path, 'w', newline='') as f:
            f.write(csv_query)

        reconstructed_query = None
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=mongo_query_delimiter, quotechar=mongo_query_quote_char)
            for row_data in reader:
                reconstructed_query = TPCHSyntheticWorkload.query_from_csv(row_data)

        # Now try executing original and reconstruction
        runtime = system.execute(query_template)
        # Runtimes are not expected to be the same.
        print("Original query run time = ", runtime)
        runtime = system.execute(reconstructed_query)
        print("Reconstructed query run time = ", runtime)
