from copy import deepcopy
import csv
from lift.case_studies.mongodb.deprecated.parsed_message import ParsedMessage
from lift.util.parsing_util import string_to_dict, dict_to_key_list
from lift.pretraining.data_source import DataSource
import logging


# TODO Deprecated -> delete?
class CombinatorialDataSource(DataSource):
    """
    Generates pre-training data from files.
    """
    def __init__(
        self,
        model_generator,
        schema
    ):
        self.model_generator = model_generator
        self.logger = logging.getLogger(__name__)
        self.states = schema.get_states_spec()
        self.actions = schema.get_actions_spec()
        self.schema_field_names = schema.get_system_spec()['key_name_to_action_index'].keys()
        self.action_lookup_dict = dict()

        self.eval_states = []
        self.eval_actions = []
        self.eval_query_keys = []

    def load_data(
        self,
        data_dir,
        generate_train_data=True,
        batch_size=250,
        max_rows=1000000,
        skip_rows=0,
        lookup_actions=False,
        create_lookup=False,
        modify_runtimes=False,
        mode=''
    ):
        data = dict(
            states={state: [] for state in self.states},
            actions={action: [] for action in self.actions},
            rewards=[],
            terminals=[],
            internals=[]
        )

        self.modify_runtimes = modify_runtimes
        data_path = data_dir + '_data.csv'
        query_path = data_dir + '_queries.csv'
        op_field_lists = self.extract_queries(max_rows, query_path, skip_rows)
        self.logger.info('Extracted queries to field lists')
        self.path = data_dir

        with open(data_path, 'r') as csvfile:
            self.logger.info('Opened path {} = '.format(data_dir))
            reader = csv.reader(csvfile, delimiter=',')
            rows_processed = 0
            batch = []
            index = 0
            for row_data in reader:
                rows_processed += 1
                if rows_processed < skip_rows:
                    continue

                op_field_list = op_field_lists[index]
                index += 1
                sort_info = string_to_dict(row_data[0])
                runtime = float(row_data[2])
                index_size = float(row_data[3])
                serialized_index = row_data[4]

                batch.append(ParsedMessage(None, None, meta_data=dict(
                    index_size=index_size,
                    runtime=runtime,
                    op_field_list=op_field_list,
                    index_name=serialized_index,
                    sort_info=sort_info)
                ))

                if rows_processed % batch_size == 0:
                    states, batch_rewards = self.model_generator.generate_state_batch(messages=batch)
                    for parsed_state in states:
                        state_meta_data = parsed_state.get_meta_data()
                        action = self.model_generator.system_to_agent_action(
                            ParsedMessage(None, None, meta_data=state_meta_data)
                        )
                        lookup_key = state_meta_data['op_list'] + '_' + str(state_meta_data['sort_info'])
                        if create_lookup and lookup_key not in self.action_lookup_dict:
                                self.action_lookup_dict[lookup_key] = action
                        if lookup_actions and lookup_key in self.action_lookup_dict:
                            # Replace actions with supervised actions
                            action = self.action_lookup_dict[lookup_key]

                        for name, batch_state in data['states'].items():
                            # single state, no dict necessary here
                            batch_state.append(parsed_state.get_value())
                        for name, batch_action in data['actions'].items():
                            batch_action.append(action[name])
                        # This is not ideal. Reward is not really state meta data
                        data['rewards'].append(state_meta_data['reward'])
                        data['terminals'].append(False)

                        if generate_train_data:
                            self.eval_states.append(dict(state=deepcopy(parsed_state.get_value())))
                            self.eval_actions.append(deepcopy(action))
                            self.eval_query_keys.append(lookup_key)

                if rows_processed % 10000 == 0 and rows_processed > 0:
                    self.logger.info('Processed rows: {}'.format(rows_processed))
                if rows_processed > max_rows:
                    break
        return data

    def extract_queries(self, max_rows, query_path, skip_rows):
        op_field_lists = []
        rows_processed = 0
        with open(query_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='%')
            for row_data in reader:
                if rows_processed < skip_rows:
                    continue
                data = ''.join(row_data)

                # Data is now a string with all query content -> dict -> fields extracted from dict
                raw_query_dict = string_to_dict(data)
                #ValueError: Empty key list, op dict was: {'status_count': {'$lt': 16}},
                op_field_list = dict_to_key_list(raw_query_dict, self.schema_field_names, data)

                # Op field list is
                op_field_lists.append(op_field_list)
                rows_processed += 1
                if rows_processed > max_rows:
                    break

        return op_field_lists

    def get_evaluation_data(self, *kwargs):
        assert self.path is not None
        return self.eval_states, self.eval_actions, self.eval_query_keys

    def reset(self):
        self.deserialized_actions = dict()
        self.deserialized_index_names = dict()
        self.path = None

        self.eval_states = []
        self.eval_actions = []
        self.eval_query_keys = []
