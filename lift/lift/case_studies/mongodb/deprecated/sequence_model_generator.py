from copy import deepcopy

from lift.case_studies.mongodb.mongo_converter import MongoConverter
import numpy as np


class SequenceConverter(MongoConverter):

    def __init__(
            self,
            experiment_config,
            schema
    ):
        super(SequenceConverter, self).__init__(experiment_config=experiment_config, schema=schema)
        # If we use action lookup, we
        # 1. run a dba run and create a dict from query fields to actions (as an approximation for the right action)
        # 2. run a default run where no actions were taken
        self.action_lookup_dict = dict()
        self.action_eos_index = self.system_spec['eos_action_index']
        self.action_sos_index = self.system_spec['sos_action_index']
        self.path = None

    def system_to_agent_action(self, message):
        index_name = message.get_meta_data()['index_name']

        # Now, we are trying to map backwards from a given index to a sequence
        # that would have constructed this index
        if index_name in self.deserialized_actions:
            return deepcopy(self.deserialized_actions[index_name])

        action_sequence = np.zeros(shape=self.actions_spec['sequence']['shape'])
        if index_name == 'none':
            action = [self.action_eos_index]
        else:
            if self.use_sort_tokens:
                action = self.system_to_agent_action_with_sort_token(index_name=index_name)
            else:
                action = self.system_to_agent_action_no_sort_token(index_name=index_name)

        elements = min(len(action_sequence), len(action))
        for i in range(elements):
            action_sequence[i] = action[i]
        action_dict = dict(sequence=action_sequence)

        self.deserialized_actions[index_name] = action_dict
        return action_dict

    def system_to_agent_action_with_sort_token(self, index_name):
        token_list = index_name.split('_')

        # Now a list of field 1 field -1
        output_sequence = [self.key_name_to_action_index[token] for token in token_list]
        output_sequence.append(self.action_eos_index)

        return output_sequence

    def system_to_agent_action_no_sort_token(self, index_name):
        token_list = index_name.split('_')
        output_sequence = []

        # Converts field_1_field_-1 into (field, 1), (field, -1) ist of tuples
        index_tuple_list = list(zip(token_list[0::2], token_list[1::2]))
        for index_tuple in index_tuple_list:
            string_index_tuple = '{}_{}'.format(index_tuple[0], index_tuple[1])
            output_sequence.append(self.key_name_to_action_index[string_index_tuple])

        output_sequence.append(self.action_eos_index)

        return output_sequence

    def agent_to_system_action(self, actions, kwargs=None):
        # Note: in combinatorial model, the expiration model receives
        # separately a key list and sort order. Here, we directly produce full action
        # Which should also ultimately be done in the combinatorial model

        # These are index tuples
        if self.use_sort_tokens:
            index_tuples = self.agent_to_system_with_sort_token(actions=actions)
        else:
            index_tuples = self.agent_to_system_no_sort_token(actions=actions)

        return index_tuples

    # A bit verbose but to avoid any mixup later
    def agent_to_system_with_sort_token(self, actions):
        # First pass, build output sequence consisting of field names and either 1 or -1 for
        # sort order
        output_sequence_tokens = []
        index_tuples = []

        prev_action_or_token = None
        for action_index in actions:
            if action_index == self.action_eos_index:
                # Previous token was field
                if prev_action_or_token in self.schema_keys:
                    index_tuples.append((prev_action_or_token, 1))
                break
            index_field_or_token = None
            if action_index in self.action_value_to_index:
                index_field_or_token = self.action_value_to_index[action_index]
                if prev_action_or_token in self.schema_keys:
                    # Both previous and current token are fields, use default sort order
                    index_tuples.append((prev_action_or_token, 1))
            elif action_index == self.action_sort_asc_token_index:
                if prev_action_or_token in self.schema_keys:
                    output_sequence_tokens.append(1)
                    index_tuples.append((prev_action_or_token, 1))
            elif action_index == self.action_sort_desc_token_index:
                if prev_action_or_token in self.schema_keys:
                    output_sequence_tokens.append(-1)
                    index_tuples.append((prev_action_or_token, -1))

            # Previous and current token both fields
            prev_action_or_token = index_field_or_token
            output_sequence_tokens.append(index_field_or_token)

        return index_tuples

    def agent_to_system_no_sort_token(self, actions):
        index_tuples = []

        # Actions is now a list of integers
        # 0 is first field, n-1 is nth field, n is eos,
        # We directly look up the tuple
        for action in actions.values():
            for step in action:
                # End token means ignore anything after.
                if step == self.action_eos_index:
                    break
                # Start token is irrelevant but may be output multiple times.
                if step == self.action_sos_index:
                    continue
                index_tuple = self.action_value_to_index[step]
                index_tuples.append(index_tuple)

        return index_tuples
