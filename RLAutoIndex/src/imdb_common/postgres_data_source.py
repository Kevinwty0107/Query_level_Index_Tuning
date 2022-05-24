import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..')) # /src for access to src/common (though not required for this file)
from imdb_common.imdb_workload import IMDBWorkload

import csv
import logging

class PostgresDataSource():
    """
    Serializes / deserializes sets of queries to or from a CSV file for resuse 
    """

    def __init__(self, workload_spec=None):
        """
        Args:
            spec (dict): workload spec required for imdbWorkload when importing (but not exporting) queries
                         de-serializing queries requires rebuilding the imdb-specific sampler which is a workload-specific closure
        """

        self.logger = logging.getLogger(__name__)

        self.workload_spec = workload_spec 

    def import_data(self, data_dir, label="", path=None):
        """
        Deserializes queries
        Calls on imdbWorkload::query_from_csv because reconstruction relies on a imdb-specific argument sampler 
        """

        if path is None:
            path = "{}/{}_queries.csv".format(data_dir, label)
        
        workload = IMDBWorkload(self.workload_spec)

        queries = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=",", quotechar="'")
            for query_csv in reader:
                query = workload.query_from_csv(query_csv) 
                queries.append(query)

        return queries

    def export_data(self, queries, data_dir, label=""):
        """
        Serializes queries
        """
        path = "{}/{}_queries.csv".format(data_dir, label)

        with open(path, 'w', newline='') as f:
            for query in queries:
                f.write(query.as_csv_row())

    def get_evaluation_data(self, **kwargs):
        pass
