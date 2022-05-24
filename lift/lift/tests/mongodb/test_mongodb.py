from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from pymongo.errors import OperationFailure

from lift.case_studies.mongodb.db_monitor import field_generator


def test_mongo_field_parser():
    # Tests recursive query parsing.

    json = {u'$or': [{u'name': u'Harvey'}, {u'user_name': u'Harvey'}]}
    fields = [u'name', u'user_name']

    keys = []
    for field in field_generator(json, fields):
        keys.append(field)

    print(keys)
    assert u'name' in keys
    assert u'user_name' in keys


def test_index_dropping(collection):
    index = [('name', 1)]
    print('Creating index')
    try:
        collection.create_index(index)
        print('Created index')
    except OperationFailure:
        print('Error creating index')
    try:
        print('Dropping index')
        collection.drop_index(index)
        print('Dropped index')
    except OperationFailure:
        print('Error dropping index')





