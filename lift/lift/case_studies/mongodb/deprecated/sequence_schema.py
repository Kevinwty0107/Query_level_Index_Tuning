from lift.case_studies.mongodb.mongo_schema import MongoSchema


class SequenceSchema(MongoSchema):

    def __init__(
        self,
        schema_config,
        experiment_config,
    ):
        super(SequenceSchema, self).__init__(schema_config, experiment_config)
        schema_spec = experiment_config['schema_spec']
        self.output_sequence_length = schema_spec['output_sequence_length']

        self.build_input_tokens()
        self.build_output_tokens()

    def build_output_tokens(self):
        # Output is a single integer sequence
        if self.use_sort_tokens:
            self.build_outputs_with_sort_tokens()
        else:
            self.build_outputs_no_sort_tokens()

        # Not RLgraph compatible due to no sequence action.
        self.actions_spec = dict(
            sequence=dict(
                type='sequence',
                shape=(self.output_sequence_length, ),
                max_sequence_length=self.output_sequence_length,
                num_actions=self.num_actions,
                embedding_size=self.schema_spec['embedding_size'],
                start_token_index=self.system_spec['sos_action_index'],
                end_token_index=self.system_spec['eos_action_index']
            )
        )

    def build_outputs_with_sort_tokens(self):
        # Here, we use extra tokens per index
        action_output_index = 0
        key_name_to_action_index = dict()
        action_value_to_index = dict()
        eos_action_index = action_output_index
        action_output_index += 1

        for key in sorted(self.schema_config.keys()):
            key_name_to_action_index[key] = action_output_index
            action_value_to_index[action_output_index] = key
            action_output_index += 1

        # Special tokens
        asc_action_token_index = action_output_index
        action_output_index += 1
        desc_action_token_index = action_output_index

        self.system_spec['key_name_to_action_index'] = key_name_to_action_index
        self.system_spec['action_value_to_field'] = action_value_to_index

        # These are used to map the integer output to a -1 or 1 for the sort tuple.
        self.system_spec['action_sort_asc_token_index'] = asc_action_token_index
        self.system_spec['action_sort_desc_token_index'] = desc_action_token_index
        self.system_spec['eos_action_index'] = eos_action_index
        action_output_index += 1
        self.system_spec['sos_action_index'] = action_output_index
        self.num_actions = action_output_index + 1

    def build_outputs_no_sort_tokens(self):
        action_output_index = 0
        # Either schema field name -> int or field name + sort order -> int
        key_name_to_action_index = dict()
        # Integer action state_value in sequence to field name or tuple
        action_value_to_index = dict()
        eos_action_index = action_output_index
        action_output_index += 1

        # Build ascending mapping
        for key in self.schema_config.keys():
            ascending_tuple_name = '{}_1'.format(key)
            key_name_to_action_index[ascending_tuple_name] = action_output_index
            action_value_to_index[action_output_index] = (key, 1)
            action_output_index += 1

        # Build descending mapping
        for key in self.schema_config.keys():
            descending_tuple_name = '{}_-1'.format(key)
            key_name_to_action_index[descending_tuple_name] = action_output_index
            action_value_to_index[action_output_index] = (key, -1)
            action_output_index += 1

        self.system_spec['key_name_to_action_index'] = key_name_to_action_index
        self.system_spec['action_value_to_index'] = action_value_to_index
        self.system_spec['eos_action_index'] = eos_action_index
        self.system_spec['sos_action_index'] = action_output_index
        self.num_actions = action_output_index + 1

