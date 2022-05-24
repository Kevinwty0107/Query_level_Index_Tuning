import numpy as np
from lift.case_studies.mongodb.query_util import random_text, random_date, random_word, random_int_array


class QueryHelper(object):

    def __init__(self, schema):
        self.schema_config = schema.schema_config

    def schema_contains(self, name):
        return name in self.schema_config

    def sample_value(self, field_name):
        field_info = self.schema_config[field_name]
        schema_type = field_info[0]

        if schema_type == 'string':
            return random_word()
        elif schema_type == 'int':
            return np.random.randint(field_info[1], field_info[2])
        elif schema_type == 'date':
            return random_date(field_info[1])
        elif schema_type == 'string_array':
            return random_text(1)
        elif schema_type == 'int_array':
            return random_int_array(1, field_info[2])
        elif schema_type == 'bool':
            return np.random.randint(low=0, high=1)
        else:
            raise ValueError("Type not supported: {}".format(schema_type))