import unittest

from lift.case_studies.mongodb import MongoPrefixRule
from lift.case_studies.mongodb.mongodb_query import MongoDBQuery


class TestMongoPrefixRule(unittest.TestCase):

    def test_rule(self):
        rule = MongoPrefixRule()

        q = MongoDBQuery(
            name='',
            tokens=[],
            query_columns=['A', 'B', 'C'],
            query_dict=dict(
               aggregation='limit',
               sort_order=[('A', 1), ('B', 1), ('C',  1)]

            )
        )
        context = []
        demo = rule.generate_demonstration(q, context)['index']
        context.append(demo)
        print('first demo = ', demo)

        prefix_q = MongoDBQuery(
            name='',
            tokens=[],
            query_columns=['A', 'B'],
            query_dict=dict(
                aggregation='limit',
                sort_order=[('A', 1), ('B', 1)]

            )
        )
        demo = rule.generate_demonstration(prefix_q, context)['index']
        print('prefix demo =', demo)
        self.assertEqual(demo, [])

        reverse_prefix_q = MongoDBQuery(
            name='',
            tokens=[],
            query_columns=['A', 'B'],
            query_dict=dict(
                aggregation='limit',
                sort_order=[('A', -1), ('B', -1)]

            )
        )
        demo = rule.generate_demonstration(reverse_prefix_q, context)['index']
        print('reverse prefix demo =', demo)
        self.assertEqual(demo, [])
        non_prefix_q = MongoDBQuery(
            name='',
            tokens=[],
            query_columns=['B', 'C'],
            query_dict=dict(
                aggregation='limit',
                sort_order=[('B', 1), ('C', 1)]

            )
        )
        demo = rule.generate_demonstration(non_prefix_q, context)['index']
        print('Non prefix demo =', demo)
        self.assertEqual(demo, non_prefix_q.full_index_from_query()['index'])
