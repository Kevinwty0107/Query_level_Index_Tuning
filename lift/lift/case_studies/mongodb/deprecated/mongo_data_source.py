import csv
from copy import deepcopy
from lift.pretraining.data_source import DataSource
import logging
import json
import numpy as np


class MongoDataSource(DataSource):
    """
    Load trace data.
    """

    def __init__(
            self,
            converter,
            schema
    ):
        self.model_generator = converter
        self.logger = logging.getLogger(__name__)
        self.states = schema.get_states_spec()
        self.actions = schema.get_actions_spec()
        self.schema_field_names = schema.get_system_spec()['schema_field_names']
        self.eval_states = []
        self.eval_actions = []
        self.index_memory = {}
        self.load_dir = None
        # TODO make configurable?
        self.next_states_required = True

    def load_data(self, data_dir, label='trace_', baseline_label='', rewards_available=False):
        """
        Loads evaluation data for pre-training.

        Args:
            data_dir (str): Directory path containing pretraining data.
            label (str): Label to identify trace.
            baseline_label (str): Name of baseline used.
            rewards_available (bool):

        Returns:
            dict: Batch of pre-training data.
        """
        actions = {}
        for name in self.actions:
            actions[name] = []
        data = dict(
            states=[],
            actions=actions,
            terminals=[],
            next_states=[],
            rewards=[]
        )
        self.load_dir = data_dir
        query_path = "{}/{}queries.json".format(data_dir, label)
        context_path = "{}/{}contexts.csv".format(data_dir, label)
        index_path = "{}/{}{}indices.csv".format(data_dir, label, baseline_label)

        # Returns list of tuples, each tuple holds query components filter, sort order, aggregation
        queries = self.load_query_tuples(query_path=query_path)
        contexts = self.load_csv(path=context_path)
        indices = self.load_csv(path=index_path)

        reward_info = None
        if rewards_available:
            reward_path = "{}/{}trace_rewards.csv".format(data_dir, label)
            reward_info = self.load_reward(path=reward_path)
        processed = 0

        episode_context = []
        for query, index in zip(queries, indices):
            # For existing indices, we need to know which sort order it is as to know
            # whether it supports the current query
            # TODO cleanup this hideous parsing -> export in better format
            existing_indices = contexts[processed]
            context_list = existing_indices[0].split(',')
            context_list = '_'.join(context_list)
            context_list = context_list.split('_')
            context_tuple_list = []
            for field, sort_order in zip(context_list[0::2], context_list[1::2]):
                context_tuple_list.append('{}_{}'.format(field, sort_order))

            episode_context.append(context_tuple_list)
            query_data = dict(
                sort_info=query[1],
                index_field_list=episode_context,
                aggregation=query[2]
            )

            # query[0] = filter
            state = self.model_generator.system_to_agent_state(
                query[0], query_data
            )

            # Split single index
            index = index[0].split(',')
            # Turn into string to enforce log format
            index = '_'.join(index)
            action = self.model_generator.system_to_agent_action(system_action=index,
                                                                 query_fields=state.get_meta_data()['query_fields'])
            if rewards_available:
                # TODO add extra file with rewards if available?
                reward = self.model_generator.system_to_agent_reward(
                    meta_data=dict(
                        # Read reward tuple
                        runtime=reward_info[processed][0],
                        index_size=reward_info[processed][1]
                    )
                )
            else:
                # Just give 0 reward, note that this matters
                # if data is continued to be used
                reward = 1.0

            data["states"].append(state.get_value())
            for name, value in action.items():
                data["actions"][name].append(value)
            data["terminals"].append(False)
            data["rewards"].append(reward)

            # Construct next state: Next context, episode context from prior qureries, next query.
            if self.next_states_required:
                next_query = queries[processed + 1] if processed < len(queries) - 1 else queries[-1]
                next_existing_indices = contexts[processed + 1] if processed < len(queries) - 1 else contexts[-1]
                next_context_list = next_existing_indices[0].split(',')
                next_context_list = '_'.join(next_context_list)
                next_context_list = next_context_list.split('_')
                next_context_tuple_list = []
                for field, sort_order in zip(next_context_list[0::2], next_context_list[1::2]):
                    next_context_tuple_list.append('{}_{}'.format(field, sort_order))

                next_episode_context = episode_context.copy()
                next_episode_context.append(next_context_tuple_list)
                next_query_data = dict(
                    sort_info=next_query[1],
                    index_field_list=next_episode_context,
                    aggregation=next_query[2]
                )

                # Update context with action
                query_data["index_field_list"] = context_tuple_list
                data["next_states"].append(self.model_generator.system_to_agent_state(
                    next_query[0], next_query_data
                ).get_value())

            self.eval_states.append(deepcopy(state.get_value()))
            self.eval_actions.append(deepcopy(action))
            processed += 1

        return self._finalize_batch(data)

    def _finalize_batch(self, data):
        """
        Finalize data by converting to arrays.

        Args:
            data (dict): Pre-training data dict.

        Returns:
            dict: Dict of numpy arrays.
        """

        batch_size = len(data["rewards"])
        # Final entry has no next state.
        i = batch_size - 1

        # Make last state terminal.
        data["terminals"][i] = True

        # Convert to single arrays.
        data["states"] = np.asarray(data["states"])
        for name in self.actions:
            # Squeeze because each action as a batch rank.
            data["actions"][name] = np.squeeze(np.asarray(data["actions"][name]))
        data["terminals"] = np.asarray(data["terminals"])
        data["next_states"] = np.asarray(data["next_states"])
        data["rewards"] = np.asarray(data["rewards"])

        return data

    @staticmethod
    def load_query_tuples(query_path):
        queries = []
        with open(query_path, 'r') as fp:
            query_dicts = json.load(fp)
            for query_dict in query_dicts:
                # Parse into raw components.
                queries.append((query_dict['query_filter'], query_dict['sort_order'], query_dict['aggregation']))

        return queries

    @staticmethod
    def load_query_dicts(query_path):
        queries = []
        with open(query_path, 'r') as fp:
            query_dicts = json.load(fp)
            for query_dict in query_dicts:
                # Parse into raw components.
                queries.append(query_dict)
        return queries

    @staticmethod
    def load_reward(path):
        reward_tuples = []
        with open(path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter='%')
            for row_data in reader:
                row_data = row_data[0].split(',')
                reward_tuples.append((float(row_data[0]), float(row_data[1])))
        return reward_tuples

    def parse_indices_to_actions(self, path):
        """
        Maps a serialized set of indices to agent actions.
        :param path: Path to indices
        :return: List of indices
        """
        actions = []
        indices = self.load_csv(path=path)
        for index in indices:
            if str(index) in self.index_memory:
                actions.append(self.index_memory[str(index)])
            else:
                index_string_list = index[0].split(',')
                if index_string_list[0] == 'none':
                    self.index_memory[str(index)] = None
                    actions.append(None)
                else:
                    index_action = []
                    # index_string = 'fieldname_1'
                    for index_string in index_string_list:
                        tuple_parts = index_string.split('_')
                        # tuple_parts = ['fieldname', '1']
                        index_action.append((tuple_parts[0], int(tuple_parts[1])))
                    self.index_memory[str(index)] = index_action
                    actions.append(index_action)
        return actions

    def get_evaluation_data(self, **kwargs):
        assert self.load_dir is not None
        return self.eval_states, self.eval_actions

    def reset(self):
        self.load_dir = None
        self.eval_states = []
        self.eval_actions = []
