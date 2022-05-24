from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from multiprocessing import Process

from pymongo import MongoClient
import logging
import re
from time import sleep
from lift.case_studies.mongodb.deprecated.log_encoder import LogEncoder
from lift.case_studies.mongodb.deprecated.log_parser import LogParser
from lift.util.parsing_util import dict_to_key_list
from lift.util.math_util import normalize
pattern = re.compile('\\[[^]]*]')


class MongoParser(LogParser):

    def __init__(
        self,
        config,
        host,
        schema,
        queue=None,
        end=0
    ):

        # Multiprocessing queue
        self.queue = queue
        self.config = config
        self.schema = schema
        self.host = host
        self.parsed_ops = ['query', 'command']

        self.logger = logging.getLogger(__name__)
        self.log_encoder = LogEncoder()
        self.end_time = end

        # Log query
        self.profile_query = {
            "ns": re.compile(r"^((?!(admin\.\$cmd|\.system|\.tmp\.)).)*$"),
            "command.profile": {"$exists": False},
            "command.collStats": {"$exists": False},
            "command.collstats": {"$exists": False},
            "command.createIndexes": {"$exists": False},
            "command.listIndexes": {"$exists": False},
            "command.cursor": {"$exists": False},
            "command.create": {"$exists": False},
            "command.dbstats": {"$exists": False},
            "command.scale": {"$exists": False},
            "command.explain": {"$exists": False},
            "command.count": {"$ne": "system.profile"},
            "op": re.compile(r"^((?!(getmore|killcursors)).)"),
        }

        # Fields to fetch
        self.projection = ['millis', 'ts', 'op', 'ns', 'query', 'updateobj', 'command', 'ninserted',
                           'ndeleted', 'nMatched', 'nreturned', 'execStats']
        self.exec_info = ['millis']

        self.system_spec = schema.get_system_spec()
        self.key_name_to_action_index = self.system_spec['key_name_to_action_index']
        self.schema_dim = self.system_spec['schema_dim']
        self.num_outputs = self.system_spec['num_outputs']
        # First half for what ops the schema touches, second for the indexing
        # +1 for number of slow queries
        self.state_size = self.system_spec['state_dim']

    def init_profiling(self):
        """
        Drops old profiling, sets configured profiling level.
        """
        #if self.reset:
        self.logger.info("Clearing system profile collection.")

        self.db.set_profiling_level(0)
        self.db.system.profile.drop()

        self.db.set_profiling_level(
            self.config['profiling_level'],
            int(self.config['slow_threshold'])
        )

    def run(self):
        """
        Runs as child process.
        :return: 
        """
        self.run_process = Process(target=self.execute, name="mongo_parser")
        self.run_process.daemon = True
        self.run_process.start()

    def execute(self):
        """
        Sends one request to the system profile collection to determine the latest slow queries
        for configured database.
        and processes.

        :return:
        """
        self.client = MongoClient(self.host, 27017)
        self.db = self.client[self.config['database']]
        self.collection = self.db[self.config['collection']]
        self.init_profiling()

        while True:
            self.logger.info('Launching cursor on database {}'.format(self.config['database']))
            cursor = self.db.system.profile.find(self.profile_query, projection=self.projection)
            # Make cursor tailable
            cursor.add_option(2)
            cursor.add_option(32)

            # Problem: if the collection is empty from beginning cursor closes immediately?
            while cursor.alive:
                for log_entry in cursor:
                    observation = self.parse_message(log_entry)
                    if observation is not None:
                        self.logger.debug("Appending entry from cursor")
                        self.queue.put(observation)
            # One idea: if the cursor dies, delete all data, restart it
            self.logger.info('Cursor died')
            #self.init_profiling()
            sleep(10)

    def parse_message(self, message):
        # print(log_entry)
        op_type = message['op']
        op_dict = None
        sort_info = None
        aggregation = None

        if op_type == 'command' and op_type in self.parsed_ops:
            if 'dbStats' in message['command']:
                return
            if 'count' in message['command']:
                aggregation = 'count'
            # Queries actually come in as commands if they use certain filters
            # self.logger.debug(log_entry)
            if 'query' in message['command']:
                op_dict = message['command']['query']
                self.logger.debug('Query in command, unencoded = ' + str(message['command']['query']))

        if op_type == 'query' and op_type in self.parsed_ops:
            query = self.log_encoder.encode(message['query']['filter']) if 'filter' in message['query'] else "{}"
            self.logger.debug('Query unencoded = ' + str(message['query']['filter']))
            if 'limit' in message['query']:
                aggregation = 'limit'
            op_dict = message['query']['filter']
            # self.logger.info(log_entry['query'])
            if 'sort' in message['query']:
                sort_info = message['query']['sort']
                aggregation = 'sort'
                query += ', sort: ' + self.log_encoder.encode(message['query']['sort'])

        elif op_type == 'insert' and op_type in self.parsed_ops:
            self.logger.debug('Insert unencoded = ' + str(message['query']))

            # Concrete query irrelevant on inserts
            query = None
            op_dict = None
        elif op_type == 'update' and op_type in self.parsed_ops:
            query = self.log_encoder.encode(message['query']) if 'query' in message else "{}"
            op_dict = message['query']

            if 'updateobj' in message:
                query += ', ' + self.log_encoder.encode(message['updateobj'])
                query += '. %s updated.' % message['nMatched']

        # Only if query not None
        if op_dict is not None:
            runtime = None
            for info in self.exec_info:
                if info in message and message[info] != {}:
                    runtime = message[info]

            index_name = None

            #TODO check recursively for input stages
            if 'execStats' in message and message['execStats'] != {}:
                # self.logger.info('execStats: {}'.format(log_entry['execStats']))

                if 'indexName' in message['execStats']['inputStage']:
                    index_name = message['execStats']['inputStage']['indexName']
                    #self.logger.info('indexName in execStats/inputStage: {}'.format(index_name))

                if 'inputStage' in message['execStats']['inputStage']:
                    if 'indexName' in message['execStats']['inputStage']['inputStage']:
                        index_name = (message['execStats']['inputStage']['inputStage']['indexName'])
                        #self.logger.info('indexName in execStats/inputStage/inputStage: {}'.format(index_name))

                    if 'inputStage' in message['execStats']['inputStage']['inputStage']:
                        if 'indexName' in message['execStats']['inputStage']['inputStage']['inputStage']:
                            index_name = (message['execStats']['inputStage']['inputStage']['inputStage']['indexName'])
                            #self.logger.info('indexName in execStats/inputStage/inputStage/inputStage/index: {}'.format(index_name))

            op_field_list = dict_to_key_list(op_dict, self.key_name_to_action_index.keys())
            # Save index size separately so we can log it
            index_size = self.get_index_size(False)
            self.logger.debug('Runtime = {} ms'.format(runtime))
            self.logger.debug('Index size = {} mb'.format(index_size))

            # State is not parsed here
            meta_data = dict(
                op_field_list=op_field_list,
                index_size=index_size,
                reward=0,
                index_name=index_name,
                sort_info=sort_info,
                aggregation=aggregation
            )

            return dict(
                request=op_dict,
                runtime=runtime,
                meta_data=meta_data
            )

    def get_index_size(self, normalized=True):
        scale = 1024 * 1024
        stats = self.db.command("dbStats")
        size = stats["indexSize"]
        size = size / scale

        # self.logger.debug("Total index size in db = " + str(size) + ' mb')
        if normalized:
            # Max size is 10000 MB for indices?
            size = normalize(size, 0, 10000)

        return size
