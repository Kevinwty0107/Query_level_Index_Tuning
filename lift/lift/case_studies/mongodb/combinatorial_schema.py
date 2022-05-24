from rlgraph.spaces import IntBox, Dict

from lift.case_studies.mongodb.mongo_schema import MongoSchema


class CombinatorialSchema(MongoSchema):
    """
    The combinatorial schema differs only in the actions structure -
    could introduce another intermediate MongoSchema class
    """
    def __init__(self, schema_config, experiment_config):
        super(CombinatorialSchema, self).__init__(schema_config, experiment_config)

        schema_spec = experiment_config['schema_spec']
        self.max_fields_per_index = schema_spec['max_fields_per_index']

        # If true, we will use separate action tokens for sort order
        # If false, we will create an ascending and a descending token per field
        self.build_input_tokens()
        self.build_output_tokens()

    def build_output_tokens(self):
        # Output is a single integer sequence
        if self.use_sort_tokens:
            raise ValueError("Sort tokens not supported in combinatorial schema.")
        else:
            self.build_outputs_no_sort_tokens()

    def build_outputs_no_sort_tokens(self):
        action_output_index = 0
        # Either schema field name -> int or field name + sort order -> int
        key_name_to_action_index = dict()
        # Integer action state_value in sequence to field name or tuple
        action_value_to_index = dict()
        noop_index = action_output_index
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

        num_outputs = action_output_index
        index_names = []

        # Dict space object from RLgraph.
        self.actions_spec = {}
        for i in range(self.max_fields_per_index):
            index_names.append('index_field{}'.format(i))
            # RLgraph spaces.
            self.actions_spec['index_field{}'.format(i)] = IntBox(
                low=0,
                high=num_outputs
            )
        self.actions_spec = Dict(self.actions_spec, add_batch_rank=True)

        # Meta data
        self.system_spec['index_names'] = index_names
        self.system_spec['num_outputs'] = int(num_outputs)
        self.system_spec['noop_index'] = noop_index
        self.system_spec['key_name_to_action_index'] = key_name_to_action_index
        self.system_spec['action_value_to_index'] = action_value_to_index
        self.system_spec['max_fields_per_index'] = self.max_fields_per_index
