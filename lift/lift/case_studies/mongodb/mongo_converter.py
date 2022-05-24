from copy import deepcopy

from lift.case_studies.mongodb.imdb_util import IMDB_SELECTIVITY
from lift.rl_model.converter import Converter
from lift.rl_model.state import State
from lift.util.parsing_util import PriorityHeap, token_generator
import logging
import numpy as np


class MongoConverter(Converter):

    def __init__(self, experiment_config, schema):
        self.logger = logging.getLogger(__name__)
        self.schema = schema
        self.experiment_config = experiment_config

        # Training data
        self.inputs = []
        self.outputs = []
        self.query_ops = []
        self.rewards = []
        self.heap = PriorityHeap()
        self.system_spec = schema.get_system_spec()
        self.state_dim = self.system_spec['state_dim']
        self.actions_spec = schema.get_actions_spec()
        self.states_spec = schema.get_states_spec()
        self.schema_keys = list(schema.schema_config.keys())

        self.state_dim = self.system_spec['state_dim']
        self.schema_fields = list(schema.schema_config.keys())
        self.index_names = self.system_spec['index_names']
        self.parser_tokens = self.system_spec['parser_tokens']
        # self.action_value_to_index = self.system_spec['action_value_to_index']
        # self.key_name_to_action_index = self.system_spec['key_name_to_action_index']

        # Special tokens.
        self.pad = self.system_spec['pad_token']
        self.index_token = self.system_spec["index_token"]

        self.state_mode = experiment_config.get('state_mode', 'default')
        if self.state_mode == 'index_net':
            self.column_slots = {}
            self.fixed_selectivity = np.zeros(shape=(len(IMDB_SELECTIVITY,)), dtype=np.float32)
            i = 0
            for col_name, selectivity in IMDB_SELECTIVITY.items():
                self.fixed_selectivity[i] = selectivity
                self.column_slots[col_name] = i
                i += 1
        # Performance objectives -> missing these is penalized in the reward.
        self.max_size = experiment_config['max_size']
        self.max_runtime = experiment_config['max_runtime']
        self.reward_penalty = experiment_config['reward_penalty']
        self.runtime_weight = experiment_config['runtime_weight']
        self.size_weight = experiment_config['size_weight']

        self.reward_mode = experiment_config.get('reward_mode', 'regularized')
        self.runtime_regularizer = experiment_config["runtime_regularizer"]
        self.size_regularizer = experiment_config["size_regularizer"]

        # Number of separate actions
        self.num_actions = len(self.actions_spec.items())
        self.input_vocabulary = self.system_spec['input_vocabulary']
        self.index_names = self.system_spec['index_names']

        # Memoizing for fast parsing
        self.deserialized_actions = {}
        self.deserialized_index_names = {}
        self.runtimes = []
        self.sizes = []

    def system_to_agent_state(self, query, system_context):
        padded_sequence = [self.input_vocabulary[self.pad]] * self.schema.input_sequence_length
        input_tokens = query.as_tokens().copy()

        for context in system_context["index_columns"]:
            append_token = False
            # Handle nested indices.
            if isinstance(context, list):
                # Only relevant context.
                # [0][0]: First list entry, first entry of tuple e.g. [(field, 1)]
                if len(context) == 1 and context[0][0] in query.query_columns:
                    field, sort_order = context[0][0], context[0][1]
                    if sort_order == 1:
                        field_token = field + "_asc_idx"
                    else:
                        field_token = field + "_desc_idx"
                    input_tokens.append(field_token)
                    append_token = True
                else:
                    # Is any sub-colum of the compound index in the query columns?
                    for index_tuple in context:
                        if index_tuple[0] in query.query_columns:
                            append_token = True
                            break

                    # If yes, append entire compound index.
                    if append_token is True:
                        for index_tuple in context:
                            field, sort_order = index_tuple[0], index_tuple[1]
                            if sort_order == 1:
                                field_token = field + "_asc_idx"
                            else:
                                field_token = field + "_desc_idx"
                            input_tokens.append(field_token)
            else:
                field, sort_order = context[0], context[1]
                if field in query.query_columns:
                    append_token = True
                    if sort_order == 1:
                        field_token = field + "_asc_idx"
                    else:
                        field_token = field + "_desc_idx"
                    input_tokens.append(field_token)
            if append_token:
                input_tokens.append(self.index_token)

        input_tokens.append(query.query_dict["aggregation"])
        # print(input_tokens)
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

    def system_to_agent_reward(self, meta_data):
        runtime = meta_data["runtime"]
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

    def get_evaluation_data(self):
        """
        Retrieves pairs of states and actions based on previously loaded data.
        """
        assert self.path is not None
        return self.inputs, self.outputs, self.query_ops

    def index_name_to_fields(self, index_name):
        """
        Extracts field names in correct order from a serialized index name
        """
        if index_name is None:
            return []
        if index_name in self.deserialized_index_names:
            return deepcopy(self.deserialized_index_names[index_name])

        fields = self.schema_fields
        count = 0
        for field in fields:
            # We find the index of fields which are in the name of the index but not in right order
            index = index_name.find(field)

            if index >= 0:
                # Field found in index
                self.heap.add_task(field, index)
                count += 1

        index_columns = [self.heap.pop_task() for _ in range(count)]
        self.deserialized_index_names[index_name] = index_columns

        return index_columns

    def reset(self):
        self.deserialized_actions = {}
        self.deserialized_index_names = {}
        self.path = None

        self.inputs = []
        self.outputs = []
        self.query_ops = []
        self.rewards = []

    def tokenize_query(self, query_filter_dict, sort_order_dict=None, index_tuple_list=None):
        """
        Extracts field-operator-state_value expressions from query.
        :return:
        """
        token_list = []
        query_columns = []
        # 1. Find tokens
        raw_tokens = list(token_generator(op_dict=query_filter_dict, tokens=self.parser_tokens))
        # Index fields = [(field, 1),(field, -1)]
        if self.use_sort_tokens:
            for token in raw_tokens:
                token_list.append(token)
                sort_direction = 1
                if token in sort_order_dict:
                    # sort order is dict of keys and their direction 1 or -1
                    sort_direction = sort_order_dict[token]
                    if sort_direction == 1:
                        token_list.append(self.state_sort_asc_token)
                    else:
                        token_list.append(self.state_sort_desc_token)
                if token in self.schema_fields:
                    query_columns.append(token)
                    if token in index_tuple_list:
                        if sort_direction == 1:
                            token_list.append(self.state_index_asc_token)
                        else:
                            token_list.append(self.state_index_desc_token)
        else:
            for token in raw_tokens:
                if token in self.schema_fields:
                    query_columns.append(token)
                    if token in sort_order_dict:
                        # sort order is dict of keys and their direction 1 or -1
                        sort_direction = sort_order_dict[token]
                        full_token = '{}_{}'.format(token, sort_direction)
                        # Look up field_name_1 or field_name_-1
                        token_list.append(full_token)

                        # If an index with the sort direction we already have exists.
                        if full_token in index_tuple_list:
                            if sort_direction == 1:
                                token_list.append(self.state_index_asc_token)
                            else:
                                token_list.append(self.state_index_desc_token)
                        # There could also be another index on the field with the other
                        # sort direction which we also need to encode because the rule based
                        # demonstrator does not care about the sort order of existing fields.
                        other_token = '{}_{}'.format(token, -sort_direction)
                        if other_token in index_tuple_list:
                            # Sort direction is -1, but the index is on 1
                            if -sort_direction == 1:
                                token_list.append(self.state_index_asc_token)
                            # Sort direction is 1, but index is on -1
                            else:
                                token_list.append(self.state_index_desc_token)
                    else:
                        # Use default sort direction:
                        full_asc_token = '{}_1'.format(token)
                        full_desc_token = '{}_-1'.format(token)

                        # No sort order on query field request
                        token_list.append(full_asc_token)

                        # Does an ascending or descending index exist?
                        if full_asc_token in index_tuple_list:
                            token_list.append(self.state_index_asc_token)
                        if full_desc_token in index_tuple_list:
                            token_list.append(self.state_index_desc_token)
                else:
                    token_list.append(token)

        return token_list, query_columns

