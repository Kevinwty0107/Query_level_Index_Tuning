from copy import deepcopy
from lift.case_studies.mongodb.mongo_converter import MongoConverter
import numpy as np


class CombinatorialConverter(MongoConverter):

    def __init__(self, experiment_config, schema):
        super(CombinatorialConverter, self).__init__(experiment_config=experiment_config, schema=schema)
        self.noop_index = self.system_spec['noop_index']
        self.max_fields_per_index = self.system_spec['max_fields_per_index']
        self.eps = 0.00001
        self.path = None

    def agent_to_system_action(self, actions, kwargs=None):
        """
        Action dict to fields. Filters index name actions.
        """
        index_tuple_list = []

        # self.logger.debug("Actions in: {}".format(actions))
        # This is an ordered list of action names 'field_1', 'field_2'
        for name in self.index_names:
            action_value = actions[name]

            # If action is packed as [action] with extra batch-dim.
            if isinstance(action_value, (list, np.ndarray)):
                action_value = action_value[0]

            if action_value != self.noop_index:
                # There could be a no-op for some of the fields
                # self.logger.debug(action_value)
                index_tuple_list.append(self.action_value_to_index[action_value])

        # self.logger.debug("Index out:  {}".format(index_tuple_list))
        return index_tuple_list

    def system_to_agent_action(self, index_name):
        # Now, we are trying to map backwards from a given index to a sequence
        # that would have constructed this index
        if index_name in self.deserialized_actions:
            return deepcopy(self.deserialized_actions[index_name])

        if index_name == 'none':
            action = dict()
            for name in self.actions_spec.keys():
                action[name] = np.asarray([self.noop_index], dtype=int)
        else:
            if self.use_sort_tokens:
                action = None  # Not implemented atm
            else:
                action = self.system_to_agent_action_no_sort_token(index_name=index_name)

        self.deserialized_actions[index_name] = action
        return action

    def system_to_agent_action_no_sort_token(self, index_name):
        token_list = index_name.split('_')

        # Converts field_1_field_-1 into (field, 1), (field, -1) ist of tuples
        index_tuple_list = list(zip(token_list[0::2], token_list[1::2]))
        action = dict()
        action_values = []
        for index_tuple in index_tuple_list:
            string_index_tuple = '{}_{}'.format(index_tuple[0], index_tuple[1])
            action_values.append(self.key_name_to_action_index[string_index_tuple])

        fill = self.max_fields_per_index - len(action_values)
        for i in range(fill):
            action_values.append(self.noop_index)

        i = 0
        # Map to separate action outputs.
        for name in sorted(self.actions_spec.keys()):
            action[name] = np.asarray([action_values[i]], dtype=int)
            i += 1

        return action
