from lift.case_studies.mongodb.imdb_synthetic_workload import IMDBSyntheticWorkload
from lift.case_studies.mongodb.imdb_util import mongo_query_delimiter, mongo_query_quote_char
from lift.pretraining.data_source import DataSource
import csv
import logging


class MongoDBDataSource(DataSource):

    def __init__(self, converter, schema):
        self.logger = logging.getLogger(__name__)

        self.converter = converter
        self.schema = schema

    def load_data(self, data_dir, label=""):
        path = "{}/{}_queries.csv".format(data_dir, label)
        self.logger.info("Restoring queries from path: {} ".format(path))

        data = []
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=mongo_query_delimiter, quotechar=mongo_query_quote_char)
            for query_csv in reader:
                reconstructed_query = IMDBSyntheticWorkload.query_from_csv(query_csv)
                data.append(reconstructed_query)

        self.logger.info("Reconstructed {} queries.".format(len(data)))
        return data

    def export_data(self, data, data_dir, label=""):
        path = "{}/{}_queries.csv".format(data_dir, label)

        with open(path, 'w', newline='') as f:
            for query in data:
                f.write(query.as_csv_row())
