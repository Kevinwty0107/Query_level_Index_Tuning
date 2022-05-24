from rlgraph.spaces import IntBox, Dict, FloatBox

from lift.case_studies.mysql.tpch_util import tpch_table_columns
from lift.rl_model.schema import Schema


class MySQLSchema(Schema):

    def __init__(self, schema_config, mode='default'):
        self.mode = mode
        self.schema_config = schema_config
        self.system_spec = {}
        self.states_spec = {}
        self.actions_spec = {}

        # If more databases than tpch become available, create lookup dict.
        # Just list tables used.
        self.tables = schema_config["tables"]

        # All columns.
        self.columns = []
        for table in self.tables:
            table_dict = tpch_table_columns[table]
            for column_name in table_dict.keys():
                self.columns.append(column_name)

        # Including default operators is not necessary if queries always have the same structure.
        self.include_default_operators = schema_config.get("include_default", False)
        self.default_operators = ["SELECT", "FROM", "WHERE"]
        self.selection_operators = ["AND", "LIKE", "IN", ">", "=", "<"]
        self.aggregation_ops = ['SORT']

        self.input_sequence_length = schema_config["input_sequence_length"]
        self.max_fields_per_index = schema_config.get("max_fields_per_index", 3)
        self.build_input_tokens()
        self.build_output_tokens()

    def build_input_tokens(self):
        self.system_spec['state_dim'] = self.input_sequence_length
        input_vocabulary = {}
        vocab_size = 0
        pad_token = 'pad'
        input_vocabulary[pad_token] = vocab_size
        vocab_size += 1

        # Columns
        for column in self.columns:
            input_vocabulary["{}_ASC".format(column)] = vocab_size
            vocab_size += 1

        for column in self.columns:
            input_vocabulary["{}_DESC".format(column)] = vocab_size
            vocab_size += 1

        # Default operators if needed.
        if self.include_default_operators:
            for operator in self.default_operators:
                input_vocabulary[operator] = vocab_size
                vocab_size += 1

        # Selection operators
        for operator in self.selection_operators:
            input_vocabulary[operator] = vocab_size
            vocab_size += 1

        for op in self.aggregation_ops:
            input_vocabulary[op] = vocab_size
            vocab_size += 1

        # Index tokens, ascending and descending:
        for key in self.columns:
            index_token = key + '_asc_idx'
            input_vocabulary[index_token] = vocab_size
            vocab_size += 1
            index_token = key + '_desc_idx'
            input_vocabulary[index_token] = vocab_size
            vocab_size += 1

        # Used to mark separate indices.
        input_vocabulary["idx"] = vocab_size
        self.states_spec = IntBox(
                low=0,
                high=vocab_size,
                shape=(self.input_sequence_length,)
        )

        self.system_spec['input_vocabulary'] = input_vocabulary
        self.system_spec['index_token'] = "idx"
        self.system_spec['pad_token'] = pad_token

        if self.mode == 'index_net':
            self.states_spec = Dict(
                sequence=IntBox(
                    low=0,
                    high=vocab_size,
                    shape=(self.input_sequence_length,)
                ),
                selectivity=FloatBox(
                    shape=(2 * len(self.columns),)
                ),
                add_batch_rank=True
            )
        else:
            self.states_spec = IntBox(
                low=0,
                high=vocab_size,
                shape=(self.input_sequence_length,)
            )

    def build_output_tokens(self):
        noop_index = 0

        index_names = []
        self.actions_spec = {}

        num_outputs = 1 + 2 * self.max_fields_per_index
        for i in range(self.max_fields_per_index):
            index_names.append('index_column{}'.format(i))
            # Use RLgraph spaces to define actions.
            self.actions_spec['index_column{}'.format(i)] = IntBox(
                low=0,
                high=num_outputs
            )
        self.actions_spec = Dict(self.actions_spec, add_batch_rank=True)

        # Meta data
        self.system_spec['index_names'] = index_names
        self.system_spec['num_outputs'] = num_outputs
        self.system_spec['noop_index'] = noop_index
        self.system_spec['max_fields_per_index'] = self.max_fields_per_index

    def get_actions_spec(self):
        return self.actions_spec

    def get_states_spec(self):
        return self.states_spec

    def get_system_spec(self):
        return self.system_spec
