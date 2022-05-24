import numpy as np
import math

import logging

from lift.case_studies.mysql.tpch_util import LINE_ITEM_SELECTIVITY
from lift.rl_model.converter import Converter
from lift.rl_model.state import State


class MySQLConverter(Converter):
    """
    Converts between RLgraph agent and MySQL server.

    This converter relies on templated query objects which expose tokenised versions of themselves
    so no parsing is necessary.
    """

    def __init__(self, experiment_config, schema):
        """

        Args:
            experiment_config (dict): Main experimental setting.
            schema (MySQLSChema): MySQlSchema object.
        """
        self.logger = logging.getLogger(__name__)

        self.schema = schema
        self.experiment_config = experiment_config

        # Reward.
        self.max_size = experiment_config['max_size']
        self.max_runtime = experiment_config['max_runtime']
        self.reward_penalty = experiment_config['reward_penalty']
        self.runtime_weight = experiment_config['runtime_weight']
        self.size_weight = experiment_config['size_weight']

        # Take square root of runtimes to smooth reward against long-running query outliers.
        self.smooth_runtime = experiment_config['smooth_runtime']
        self.runtime_regularizer = experiment_config["runtime_regularizer"]
        self.size_regularizer = experiment_config["size_regularizer"]
        self.reward_mode = experiment_config.get('reward_mode', 'regularized')

        self.state_mode = experiment_config.get('state_mode', 'default')
        if self.state_mode == 'index_net':
            self.column_slots = {}
            self.fixed_selectivity = np.zeros(shape=(len(LINE_ITEM_SELECTIVITY,)), dtype=np.float32)
            i = 0
            for col_name, selectivity in LINE_ITEM_SELECTIVITY.items():
                self.fixed_selectivity[i] = selectivity
                self.column_slots[col_name] = i
                i += 1

        self.system_spec = schema.get_system_spec()
        self.actions_spec = schema.get_actions_spec()
        self.max_columns_per_index = len(self.actions_spec)
        self.index_token = self.system_spec["index_token"]
        # Special tokens.
        self.pad = self.system_spec['pad_token']
        self.input_vocabulary = self.system_spec['input_vocabulary']
        self.noop_index = self.system_spec['noop_index']

        self.runtimes = []
        self.sizes = []

    def system_to_agent_state(self, query, system_context):
        """
        Converts a query and a system context to an agent state.

        Args:
            query (SQLQuery): Templated SQLQuery object.
            system_context (dict): Context.

        Returns:
            State: State object.
        """
        padded_sequence = [self.input_vocabulary[self.pad]] * self.schema.input_sequence_length

        # Query provides a tokenised list.
        input_tokens = query.as_tokens().copy()

        # Consider context but only any indices where at least one field matches
        # query columns -> not relevant otherwise.
        # print("Tokens before context =", input_tokens)
        # print("context = ", system_context["index_columns"])
        for context in system_context["index_columns"]:
            append_token = False
            # Handle nested indices.
            if isinstance(context, list):
                # Only relevant context.
                # [0][0]: First list entry, first entry of tuple e.g. [(field, 1)]
                if len(context) == 1 and context[0][0] in query.query_columns:
                    field, sort_order = context[0][0], context[0][1]
                    if sort_order == "ASC":
                        field_token = field + "_asc_idx"
                    else:
                        field_token = field + "_desc_idx"
                    input_tokens.append(field_token)
                    append_token = True
                else:
                    # Is any sub-column of the compound index in the query columns?
                    for index_tuple in context:
                        if index_tuple[0] in query.query_columns:
                            append_token = True
                            break

                    # If yes, append entire compound index.
                    if append_token is True:
                        for index_tuple in context:
                            field, sort_order = index_tuple[0], index_tuple[1]
                            if sort_order == "ASC":
                                field_token = field + "_asc_idx"
                            else:
                                field_token = field + "_desc_idx"
                            input_tokens.append(field_token)
            else:
                field, sort_order = context[0], context[1]
                if field in query.query_columns:
                    append_token = True
                    if sort_order == "ASC":
                        field_token = field + "_asc_idx"
                    else:
                        field_token = field + "_desc_idx"
                    input_tokens.append(field_token)
            if append_token:
                input_tokens.append(self.index_token)

        # print("Tokens after context =", input_tokens)
        # Debug token length -> Potentially need longer input sequence.
        # print(len(input_tokens))
        # Lookup.
        # print("input tokens = ", input_tokens)
        # print("Raw tokens = ", input_tokens)
        indexed_input_sequence = [self.input_vocabulary[input_word] for input_word in input_tokens]

        # Fill up sequence array up to max length.
        elements = min(len(indexed_input_sequence), len(padded_sequence))
        for i in range(elements):
            padded_sequence[i] = indexed_input_sequence[i]

        # self.logger.info("State sequence = {}".format(padded_sequence))
        if self.state_mode == 'index_net':
            multi_hot_columns = np.zeros_like(self.fixed_selectivity)
            # 1.0 wherever 1.0
            for column in query.query_columns:
                col_index = self.column_slots[column]
                multi_hot_columns[col_index] = 1.0
            state = dict(
                sequence=np.asarray(padded_sequence),
                selectivity=np.concatenate([self.fixed_selectivity, multi_hot_columns])
            )
            return State(value=state, meta_data=dict(query_columns=query.query_columns))
        else:
            return State(value=np.asarray(padded_sequence), meta_data=dict(query_columns=query.query_columns))

    def system_to_agent_action(self, system_action, query):
        index_columns = system_action["index"]
        action = {}

        if len(index_columns) == 0:
            for name in self.actions_spec.keys():
                action[name] = np.asarray([self.noop_index], dtype=int)

        action_values = []

        # Build the order in which columns appear in the query.
        column_order = {}
        i = 1
        for column in query.query_columns:
            column_order[column] = i
            i += 1

        #  [P_PART, P_BRAND]
        # Find position of index column in input columns.
        for index_tuple in index_columns:
            # Find position of column in input.
            input_position = column_order[index_tuple[0]]
            # E.g. input position index is 2 for second field
            # Now map this to the correct offset via sort order
            if index_tuple[1] == "ASC":
                # e.g. input position of first field = 1, ascending -> action state_value is 1
                # e.g. input position of second field = 2, ascending -> action state_value is 3
                action_values.append(1 + (input_position - 1) * 2)
            elif index_tuple[1] == "DESC":
                # e.g. input position of first field = 1, descending -> action state_value is 2
                action_values.append(1 + (input_position - 1) * 2 + 1)

        # Fill up with no-ops.
        fill = self.max_columns_per_index - len(action_values)
        for i in range(fill):
            action_values.append(self.noop_index)

        i = 0
        # Map to action outputs.
        for name in sorted(self.actions_spec.keys()):
            action[name] = np.asarray([action_values[i]], dtype=int)
            i += 1

        return action

    def agent_to_system_action(self, actions, meta_data=None):
        """
        Action dict to fields. Filters index name actions.

        Args:
            actions (dict): Actions dict.
            meta_data (dict): Conversion info.
        """
        index_columns_tuples = []
        columns_in_index = []
        input_fields = meta_data["query_columns"]

        # MySQL indices are not sort combined.
        num_input_fields = len(input_fields)
        for name in self.actions_spec.keys():
            # E.g. for 2 actions [0, column_1, column_2]
            # = [0, 1, 2]
            # input fields = [column_3, column_4]
            # actions: 0 2
            # noop, column_4
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
                    column = input_fields[action_input_field - 1]
                    if column not in columns_in_index:
                        columns_in_index.append(column)
                        if action_value % 2 == 1:
                            # Look up the input column the action is referring to.
                            index_columns_tuples.append((column, "ASC"))
                        else:
                            index_columns_tuples.append((column, "DESC"))

        # Default to "ASC" if only single column (makes no difference in execution), so we
        # do not create COLUMN_ASC and COLUMN_DESC both.
        if len(index_columns_tuples) == 1:
            index_columns_tuples[0] = (index_columns_tuples[0][0], "ASC")
        return index_columns_tuples

    def system_to_agent_reward(self, meta_data):
        runtime = meta_data["runtime"]
        if self.smooth_runtime:
            runtime = math.sqrt(runtime)
        index_size = meta_data["index_size"]
        reward = 0
        if self.reward_mode == 'additive':
            reward = - (self.runtime_weight * runtime) - (self.size_weight * index_size)
            if runtime > self.max_runtime:
                reward -= self.reward_penalty
            if index_size > self.max_size:
                reward -= self.reward_penalty
        elif self.reward_mode == 'regularized':
            if runtime > 0:
                # Regularise reward so well behaved for very  small runtimes.
                reward += self.runtime_weight / (runtime + self.runtime_regularizer)
            if index_size > 0:
                reward += self.size_weight / (index_size + self.size_regularizer)
        elif self.reward_mode == 'normalized_additive':
            self.runtimes.append(runtime)
            self.sizes.append(index_size)
            # Only begin normalizing after first episode.
            if len(self.runtimes) > 20:
                runtime = (runtime - np.mean(self.runtimes) / (np.std(self.runtimes) + 0.00001))
                index_size = (index_size - np.mean(self.sizes) / (np.std(self.sizes) + 0.00001))
            reward = - (self.runtime_weight * runtime) - (self.size_weight * index_size)
        return reward

