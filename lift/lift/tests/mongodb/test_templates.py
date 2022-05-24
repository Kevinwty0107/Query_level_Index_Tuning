import json


def test_create_state_template():
    schema_dim = 0
    schema_template = {}
    schema_config = json.loads('/Users/Michael/Documents/deep_config/DeepReconfig/lift/case_studies/conf/schema.json')

    # Field name to action index
    for key in schema_config:
        # Set index of key
        print('Before key = {}'.format(key))
        k = key[0]
        print('After k = {}'.format(k))

        schema_template[k] = schema_dim

        schema_dim += 1