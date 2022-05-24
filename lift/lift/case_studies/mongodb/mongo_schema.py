from rlgraph.spaces import IntBox, Dict, FloatBox
from lift.rl_model.schema import Schema


class MongoSchema(Schema):

    def __init__(self, schema_config, schema_spec, mode='default'):
        self.mode = mode
        self.schema_config = schema_config
        self.system_spec = {}
        self.states_spec = {}
        self.actions_spec = {}

        self.input_sequence_length = schema_spec['input_sequence_length']
        self.logical_ops = ['$and', '$or', '$nor', '$not']
        self.comparison_ops = ['$eq', '$gt', '$lt', '$gte', '$lte', '$nin']
        self.aggregation_ops = ['sort', 'limit', 'count']
        self.schema_spec = schema_spec

        # Often only using 1 collection.
        self.collections = [schema_spec["collections"]] if isinstance(schema_spec["collections"], str) == 1 else \
            schema_spec["collections"]

        self.collection_to_schema = {self.collections[0]: schema_config}
        self.init()

    def init(self):
        self.system_spec = {'index_names': 'sequence_index'}
        # Array ops
        is_array_field = {}
        schema_field_names = []
        for key, description in self.schema_config.items():
            field_type = description[0]
            schema_field_names.append(key)
            if field_type in ['string_array']:
                is_array_field[key] = True
            else:
                is_array_field[key] = False

        self.system_spec['is_array_field'] = is_array_field
        self.system_spec['schema_field_names'] = schema_field_names
        self.build_input_tokens()

    def get_actions_spec(self):
        return self.actions_spec

    def get_states_spec(self):
        return self.states_spec

    def get_system_spec(self):
        return self.system_spec

    def build_input_tokens(self):
        self.system_spec['state_dim'] = self.input_sequence_length

        input_vocabulary = {}
        vocab_size = 0
        pad_token = 'pad'
        input_vocabulary['pad'] = vocab_size
        parser_tokens = []
        # Input tokens = fields + ops + special token
        vocab_size += 1

        # Build ascending mapping
        for key in self.schema_config.keys():
            ascending_key = '{}_1'.format(key)
            # NOTE: using key here, because we parse the query filter dict which contains
            # the field names, not field_name_1. The vocabulary look up is then done with field_1
            parser_tokens.append(key)
            input_vocabulary[ascending_key] = vocab_size
            vocab_size += 1

        # Build descending mapping
        for key in self.schema_config.keys():
            ascending_key = '{}_-1'.format(key)
            input_vocabulary[ascending_key] = vocab_size
            vocab_size += 1

        # Map ops to vocabulary.
        for op in self.logical_ops:
            parser_tokens.append(op)
            input_vocabulary[op] = vocab_size
            vocab_size += 1
        for op in self.comparison_ops:
            parser_tokens.append(op)
            input_vocabulary[op] = vocab_size
            vocab_size += 1
        for op in self.aggregation_ops:
            parser_tokens.append(op)
            input_vocabulary[op] = vocab_size
            vocab_size += 1

        # Index tokens, ascending and descennding:
        for key in self.schema_config.keys():
            index_token = key + '_asc_idx'
            input_vocabulary[index_token] = vocab_size
            vocab_size += 1
            index_token = key + '_desc_idx'
            input_vocabulary[index_token] = vocab_size
            vocab_size += 1

        input_vocabulary["idx"] = vocab_size

        self.system_spec['parser_tokens'] = parser_tokens
        self.system_spec['input_vocabulary'] = input_vocabulary
        self.system_spec['index_token'] = "idx"

        # This should be zero
        self.system_spec['pad_token'] = pad_token

        if self.mode == 'index_net':
            self.states_spec = Dict(
                sequence=IntBox(
                    low=0,
                    high=vocab_size,
                    shape=(self.input_sequence_length,)
                ),
                selectivity=FloatBox(
                    shape=(2 * len(self.schema_config),)
                ),
                add_batch_rank=True
            )
        else:
            self.states_spec = IntBox(
                low=0,
                high=vocab_size,
                shape=(self.input_sequence_length,)
            )