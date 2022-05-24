import logging
import time

from lift.case_studies.mongodb.imdb_util import imdb_collection_info
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from lift.case_studies.mongodb.templated_query import TemplatedQuery
from lift.rl_model.system_environment import SystemEnvironment


class MongoDBSystemEnvironment(SystemEnvironment):

    def __init__(
        self,
        experiment_config=None,
        host='localhost'
    ):
        super(MongoDBSystemEnvironment, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.query_helper = None
        self.index_set = set()

        self.config = experiment_config
        self.host = host
        # This does not need to be in a separate init function because there is no
        # separate process for the model, thus no fork problem (not forksafe client).
        self.client = MongoClient(self.host, 27017)
        self.db = self.client[self.config['database']]
        self.collection = self.db[self.config['collection']]
        self.indices_created = 0

    def act(self, action):
        if isinstance(action, dict):
            action = action["index"]
        start = time.monotonic()
        if action:
            if str(action) in self.index_set:
                self.logger.info("Action already in index set, not executing.")
            elif self.is_noop(action):
                self.logger.info("Action is no-op, not executing.")
            else:
                try:
                    # self.logger.info("Creating index: {}".format(action))
                    self.collection.create_index(action)
                    self.index_set.add(str(action))
                    self.indices_created += 1
                except Exception as e:
                    self.logger.info('Failed index {}: {}'.format(action, e))
                    time.sleep(60)
                    self.client = MongoClient(self.host, 27017)
                    self.db = self.client[self.config['database']]
                    self.collection = self.db[self.config['collection']]
                    self.collection.create_index(action)
                    self.index_set.add(str(action))
                    self.indices_created += 1
        return time.monotonic() - start

    def system_status(self):
        # Index size in GB.
        scale = 1024 * 1024 * 1024
        try:
            stats = self.db.command("dbStats")
            size = stats["indexSize"]
            size = size / scale
            current_config = self.collection.index_information()
            return size, self.indices_created, current_config
        except Exception as e:
            self.logger.info('Failed system status = {}'.format(e))
            time.sleep(60)
            # Retry
            stats = self.db.command("dbStats")
            size = stats["indexSize"]
            size = size / scale
            current_config = self.collection.index_information()
            return size, self.indices_created, current_config

    def execute(self, query):
        """
        Executes MongoDB query object.

        Args:
            query (MongoDBQuery:

        Returns:
            float: Query execution time.
        """
        try:
            return query.sample_fn(self.collection)
        except Exception as e:
            self.logger.info('Failed query execution, waiting for retry.')
            time.sleep(60)
            return query.sample_fn(self.collection)

    def explain(self, query):
        return query.explain(self.collection)

    def is_noop(self, action):
        if not action:
            self.logger.info("Action is noop = {}".format(action))
            return True
        # Check if action spans multiple arrays -> not allowed
        array_fields = 0
        for index_tuple in action:
            if imdb_collection_info[index_tuple[0]][0] == "string_array":
                array_fields += 1
        if array_fields > 1:
            return True
        else:
            return False

    def make_executable(self, queries, sample_values=False):
        """
        Maps a set of serialized queries to a set of executable functions
        :param sample_values: If queries should sample state_value on each call.
        :param queries: List of query dicts.

        :return: List of executable query objects.
        """
        executable_queries = []
        for query in queries:
            executable_queries.append(TemplatedQuery(
                collection=self.collection,
                query_dict=query,
                sample_values=sample_values,
                query_helper=self.query_helper
            ))

        return executable_queries

    def reset(self):
        try:
            self.collection.drop_indexes()
        except Exception as e:
            self.logger.info('Failed dropping indices, waiting for retry.')
            time.sleep(60)
            self.collection.drop_indexes()
        self.index_set.clear()
        self.indices_created = 0

