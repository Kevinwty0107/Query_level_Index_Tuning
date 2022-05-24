from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime

from pymongo import MongoClient
from pymongo.errors import OperationFailure
import logging
import time

from lift.rl_model.system_environment import SystemEnvironment


class CombinatorialExpirationModel(SystemEnvironment):
    """
    Indices are created and renewed when an action is selected. Indices automatically expire if not
    selected again.
    """

    def __init__(
        self,
        experiment_config=None,
        schema=None,
        agent_config=None,
        host='localhost',
        queue=None,
        model_generator=None
    ):
        SystemEnvironment.__init__(self)

        self.model_generator = model_generator
        self.queue = queue
        self.logger = logging.getLogger(__name__)
        self.config = experiment_config
        self.host = host
        self.sort_order = experiment_config['sort_order']
        self.eps = 0.0001

        self.ttl = experiment_config['ttl']
        self.agent_config = agent_config
        self.key_name_to_action_index = schema.get_system_spec()['key_name_to_action_index']
        self.schema_dim = schema.get_system_spec()['schema_dim']

        self.index_set = set()
        self.expirations = dict()
        self.index_count = 0

    def init_model(self):
        self.client = MongoClient(self.host, 27017)
        self.db = self.client[self.config['database']]
        self.collection = self.db[self.config['collection']]

        self.load_indices()

    def act(self, action):
        # Convert action to index fields
        key_list = self.model_generator.agent_to_system_action(action)
        if self.sort_order:
            index = self.keylist_to_index(key_list, action['sort_order'])
        else:
            index = self.keylist_to_index(key_list)

        # Can have multiple indices with different sort orders on same field -> sort order needs to be serialized
        key_string = str(index)
        runtime = -1

        if key_string in self.index_set:
            self.logger.info("Index exists for keys: " + key_string )
            existed = True
        else:
            existed = False
            self.logger.info("Creating index on:" + str(index))
            start = time.time()
            try:
                self.collection.create_index(index, background=True)
            except OperationFailure:
                self.logger.info('Failed index {}'.format(index))
            runtime = time.time() - start
            self.index_count += 1
            self.index_set.add(key_string)

        # Extend expiration independent from whether this index existed
        self.extend_expiration(key_string, index)
        self.check_expirations()

        return dict(
            index_creation_info=(str(datetime.datetime.now()), runtime, existed, self.index_count),
            existed=existed
        )

    def get_info(self):
        """
        Logs information on all current indices.
        """
        self.logger.info("Retrieving index information on collection:" + str(self.config.collection))
        index_info = self.collection.index_information()
        for key, value in index_info.items():
            self.logger.info("Index on key" + str(key) + ", info = " + str(value))

    def observe_system(self, batch_size=0):
        ops = []
        while not self.queue.empty():
            observation = self.queue.get()
            if observation is not None:
                ops.append(observation)

        return ops

    def generate_state_batch(self, observations):
        return self.model_generator.generate_state_batch(observations)

    def check_expirations(self):
        """
        Execute post action logic. For each action, we update the expiration time stamp.
        We check any expired timestamps and remove the respective indices.
        """
        removed = []
        for key, expiration_item in self.expirations.items():
            if 0 < expiration_item[0] < time.time():
                # Key is now 'field1' or 'field1_field2'
                index = expiration_item[1]
                # Keylist is now ['field1', 'field2']
                self.drop_index(index)
                self.expirations[key] = -1
                self.index_count -= 1
                self.index_set.remove(key)

    def keylist_to_index(self, keys, sort_action=None):
        """
        Convert keys to index-creation list of tuples.

        :param keys: List of keys to convert to index creation tuples.
        :param sort_action: Sort action
        :return: List of tuples for an index identifier
        """
        index = []
        if sort_action is not None:
            i = 0
            sort_order = self.model_generator.action_to_sort_order[str(sort_action)]
            for key in keys:
                # Use provided order
                #TODO if array type, skip
                index.append((key, sort_order[i]))
                i += 1
        else:
            for key in keys:
                # Use default order
                index.append((key, 1))

        return index

    def extend_expiration(self, key, index):
        """
        Extend the expiration of key by index ttl.
        :param key: Key-string
        """
        self.expirations[key] = [time.time() + self.ttl, index]

    def drop_index(self, keylist):
        index = self.keylist_to_index(keylist)
        try:
            self.logger.info("Dropping index on:" + str(keylist))
            self.collection.drop_index(index)
            self.logger.info('Dropped index')
        except OperationFailure:
            self.logger.error('Error dropping index:')
            self.logger.error(index)

    def system_status(self):
        return self.collection.index_information()

    def load_indices(self):
        """
        Preload index information.
        """
        index_info = self.collection.index_information()
        for key, value in index_info.items():
            if key != '_id_':
                #TODO keystring would be 'field_1', not 'field'
                # This works to just note the number of indices
                self.index_count += 1
                self.index_set.add(key)

    def is_noop(self, action):
        return (action['index_field0'] + action['index_field1']) == 0



