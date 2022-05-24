from rlgraph.spaces import IntBox, Dict

from lift.case_studies.mongodb.mongo_schema import MongoSchema


class FieldPositionSchema(MongoSchema):
    """
    The field position schema doesnt output field names but just whether a certain
    input in the position needs an index and if so what direction.
    """
    def __init__(self, schema_config, schema_spec, mode='default'):
        super(FieldPositionSchema, self).__init__(schema_config=schema_config,
                                                  schema_spec=schema_spec,
                                                  mode=mode)

        self.max_fields_per_index = schema_spec['max_fields_per_index']

        self.build_input_tokens()
        self.build_output_tokens()

    def build_output_tokens(self):
        noop_index = 0

        index_names = []
        self.actions_spec = {}

        # 1 for noop, asc + desc for every positional field
        # E.g. for 2 actions [0, first_field_1, first_field_-1, second_field_1, second_field_-1]
        # = [0, 1, 2 3, 4]
        # input fields = [field3, field4]
        # Assume index given is field_4_-1
        # Second input position -> 2.
        # Action index offset = 1 + (input_position - 1) * 2
        # Action index = offset + (0 if ascending else 1)
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
