
import yaml
import re

from lift.util.parsing_util import token_generator

schema_template = {
    "karma": 1,
    "name": 1,
    "user_name": 1,
    "user_role": 1,
    "registration_date": 1,
    "follower_count": 1,
    "status_hashtags": 1
}

def test_json_loading():
    a = "{'registration_date': {'gt': datetime.datetime(2016, 4, 27, 23, 8, 33, 481000)},'follower_count': {'lt': 91}}"
    s = a.replace("'", "\"")

    op_dict = yaml.load(s)

    print(op_dict)
    keys = []
    for field in token_generator(op_dict, schema_template.keys()):
        # self.logger.debug("field in dict to key = " + str(field))
        keys.append(field)

    print(keys)

def test_ignore_values():
    q = "{'status_hashtags': ['name']}"
    s = q.replace("'", "\"")

    pattern = re.compile('\\[[^]]*]')
    result = pattern.sub("replace", s)
    print(result)
    op_dict = yaml.load(result)

    print(op_dict)
    keys = []
    for field in token_generator(op_dict, schema_template.keys()):
        # self.logger.debug("field in dict to key = " + str(field))
        keys.append(field)

    print(keys)

