from copy import deepcopy
from lift.case_studies.mongodb.mongo_converter import MongoConverter
import numpy as np


class FieldPositionConverter(MongoConverter):

    def __init__(self, experiment_config, schema):
        super(FieldPositionConverter, self).__init__(experiment_config=experiment_config, schema=schema)
        self.noop_index = self.system_spec['noop_index']
        self.max_fields_per_index = self.system_spec['max_fields_per_index']
        self.eps = 0.00001
        self.path = None

    def agent_to_system_action(self, actions, meta_data=None):
        """
        Action dict to fields. Filters index name actions.

        Args:
            actions (dict): Actions dict.
            meta_data (dict): Conversion info.
        """
        index_tuple_list = []
        input_fields = meta_data['query_columns']

        num_input_fields = len(input_fields)
        index_fields = []
        for name in self.index_names:
            # E.g. for 2 actions [0, first_field_1, first_field_-1, second_field_1, second_field_-1]
            # = [0, 1, 2 3, 4]
            # input fields = [field3, field4]
            # actions: 0 3
            # noop, field_4_1
            action_value = actions[name]

            # If action is packed as [action] with extra batch-dim.
            if isinstance(action_value, (list, np.ndarray)):
                action_value = action_value[0]
            if action_value != self.noop_index:
                # E.g. output 3 gives field 2, output 1 gives field 1
                action_input_field = int((action_value - 1) / 2) + 1

                # Now, only if input has this field length:
                if action_input_field <= num_input_fields:
                    # Odds are ascending, even values descending
                    index_field = input_fields[action_input_field - 1]

                    if index_field not in index_fields:
                        # Do not allow reusing same column.
                        index_fields.append(index_field)
                        if action_value % 2 == 1:
                            # Look up the input field the action is referring to.
                            index_tuple_list.append((index_field, 1))
                        else:
                            index_tuple_list.append((index_field, -1))

        # Sanitize: Single-column indices always ascending to avoid doubly created.
        if len(index_tuple_list) == 1:
            index_tuple_list[0] = (index_tuple_list[0][0], 1)
        return index_tuple_list

    def system_to_agent_action(self, system_action, query):
        if isinstance(system_action, dict):
            system_action = system_action["index"]
        query_columns = query.query_columns
        if system_action == 'none' or system_action == []:
            action = {}
            for name in self.actions_spec.keys():
                action[name] = np.asarray(self.noop_index, dtype=int)
        else:
            action = self.system_to_agent_action_no_sort_token(index_tuple_list=system_action, query_columns=query_columns)

        return action

    def system_to_agent_action_no_sort_token(self, index_tuple_list, query_columns):
        action = {}
        action_values = []
        query_field_dict = {}
        i = 1
        for field in query_columns:
            query_field_dict[field] = i
            i += 1
        for index_tuple in index_tuple_list:
            # Fetch input position of field via field name
            input_position = query_field_dict[index_tuple[0]]

            # E.g. input position index is 2 for second field
            # Now map this to the correct offset via sort order
            if int(index_tuple[1]) == 1:
                # e.g. input position of first field = 1, ascending -> action state_value is 1
                # e.g. input position of second field = 2, ascending -> action state_value is 3
                action_values.append(1 + (input_position - 1) * 2)
            elif int(index_tuple[1]) == -1:
                # e.g. input position of first field = 1, descending -> action state_value is 2
                action_values.append(1 + (input_position - 1) * 2 + 1)

        fill = self.max_fields_per_index - len(action_values)
        for i in range(fill):
            action_values.append(self.noop_index)

        i = 0
        # Map to separate action outputs.
        for name in sorted(self.actions_spec.keys()):
            action[name] = np.asarray([action_values[i]], dtype=int)
            i += 1

        return action
