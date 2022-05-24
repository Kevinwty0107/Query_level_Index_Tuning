import unittest


class TestIndexEvaluation(unittest.TestCase):

    expected = {
        "endYear_1": 1,
        "isOriginalTitle_1": 1,
        "language_1_types_-1": 2,
        "startYear_1_language_-1": 1,
        "title_1": 1
    }

    # Raw dumps.
    all_stats = [
        ################
        {'queryPlanner': {'plannerVersion': 1, 'namespace': 'imdb.imdb_all', 'indexFilterSet': False,
                          'parsedQuery': {'endYear': {'$lt': 1966}}, 'winningPlan': {'stage': 'FETCH',
                                                                                     'inputStage': {'stage': 'IXSCAN',
                                                                                                    'keyPattern': {
                                                                                                        'endYear': 1},
                                                                                                    'indexName': 'endYear_1',
                                                                                                    'isMultiKey': False,
                                                                                                    'multiKeyPaths': {
                                                                                                        'endYear': []},
                                                                                                    'isUnique': False,
                                                                                                    'isSparse': False,
                                                                                                    'isPartial': False,
                                                                                                    'indexVersion': 2,
                                                                                                    'direction': 'forward',
                                                                                                    'indexBounds': {
                                                                                                        'endYear': [
                                                                                                            '[-inf.0, 1966)']}}},
                          'rejectedPlans': []},
         'executionStats': {'executionSuccess': True, 'nReturned': 2328, 'executionTimeMillis': 4,
                            'totalKeysExamined': 2328, 'totalDocsExamined': 2328,
                            'executionStages': {'stage': 'FETCH', 'nReturned': 2328, 'executionTimeMillisEstimate': 10,
                                                'works': 2329, 'advanced': 2328, 'needTime': 0, 'needYield': 0,
                                                'saveState': 18, 'restoreState': 18, 'isEOF': 1, 'invalidates': 0,
                                                'docsExamined': 2328, 'alreadyHasObj': 0,
                                                'inputStage': {'stage': 'IXSCAN', 'nReturned': 2328,
                                                               'executionTimeMillisEstimate': 0, 'works': 2329,
                                                               'advanced': 2328, 'needTime': 0, 'needYield': 0,
                                                               'saveState': 18, 'restoreState': 18, 'isEOF': 1,
                                                               'invalidates': 0, 'keyPattern': {'endYear': 1},
                                                               'indexName': 'endYear_1', 'isMultiKey': False,
                                                               'multiKeyPaths': {'endYear': []}, 'isUnique': False,
                                                               'isSparse': False, 'isPartial': False, 'indexVersion': 2,
                                                               'direction': 'forward',
                                                               'indexBounds': {'endYear': ['[-inf.0, 1966)']},
                                                               'keysExamined': 2328, 'seeks': 1, 'dupsTested': 0,
                                                               'dupsDropped': 0, 'seenInvalidated': 0}},
                            'allPlansExecution': []},
         'serverInfo': {'host': 'kikyo.cl.cam.ac.uk', 'port': 27017, 'version': '4.0.9',
                        'gitVersion': 'fc525e2d9b0e4bceff5c2201457e564362909765'}, 'ok': 1.0},

        ################
        {'queryPlanner': {'plannerVersion': 1, 'namespace': 'imdb.imdb_all', 'indexFilterSet': False,
                          'parsedQuery': {'isOriginalTitle': {'$eq': 0}}, 'winningPlan': {'stage': 'FETCH',
                                                                                          'inputStage': {
                                                                                              'stage': 'IXSCAN',
                                                                                              'keyPattern': {
                                                                                                  'isOriginalTitle': 1},
                                                                                              'indexName': 'isOriginalTitle_1',
                                                                                              'isMultiKey': False,
                                                                                              'multiKeyPaths': {
                                                                                                  'isOriginalTitle': []},
                                                                                              'isUnique': False,
                                                                                              'isSparse': False,
                                                                                              'isPartial': False,
                                                                                              'indexVersion': 2,
                                                                                              'direction': 'forward',
                                                                                              'indexBounds': {
                                                                                                  'isOriginalTitle': [
                                                                                                      '[0, 0]']}}},
                          'rejectedPlans': [{'stage': 'FETCH',
                                             'inputStage': {'stage': 'IXSCAN', 'keyPattern': {'isOriginalTitle': -1},
                                                            'indexName': 'isOriginalTitle_-1', 'isMultiKey': False,
                                                            'multiKeyPaths': {'isOriginalTitle': []}, 'isUnique': False,
                                                            'isSparse': False, 'isPartial': False, 'indexVersion': 2,
                                                            'direction': 'forward',
                                                            'indexBounds': {'isOriginalTitle': ['[0, 0]']}}}]},
         'executionStats': {'executionSuccess': True, 'nReturned': 3230239, 'executionTimeMillis': 3701,
                            'totalKeysExamined': 3230239, 'totalDocsExamined': 3230239,
                            'executionStages': {'stage': 'FETCH', 'nReturned': 3230239,
                                                'executionTimeMillisEstimate': 3230, 'works': 3230240,
                                                'advanced': 3230239, 'needTime': 0, 'needYield': 0, 'saveState': 25237,
                                                'restoreState': 25237, 'isEOF': 1, 'invalidates': 0,
                                                'docsExamined': 3230239, 'alreadyHasObj': 0,
                                                'inputStage': {'stage': 'IXSCAN', 'nReturned': 3230239,
                                                               'executionTimeMillisEstimate': 1190, 'works': 3230240,
                                                               'advanced': 3230239, 'needTime': 0, 'needYield': 0,
                                                               'saveState': 25237, 'restoreState': 25237, 'isEOF': 1,
                                                               'invalidates': 0, 'keyPattern': {'isOriginalTitle': 1},
                                                               'indexName': 'isOriginalTitle_1', 'isMultiKey': False,
                                                               'multiKeyPaths': {'isOriginalTitle': []},
                                                               'isUnique': False, 'isSparse': False, 'isPartial': False,
                                                               'indexVersion': 2, 'direction': 'forward',
                                                               'indexBounds': {'isOriginalTitle': ['[0, 0]']},
                                                               'keysExamined': 3230239, 'seeks': 1, 'dupsTested': 0,
                                                               'dupsDropped': 0, 'seenInvalidated': 0}},
                            'allPlansExecution': [
                                {'nReturned': 101, 'executionTimeMillisEstimate': 0, 'totalKeysExamined': 101,
                                 'totalDocsExamined': 101, 'executionStages': {'stage': 'FETCH', 'nReturned': 101,
                                                                               'executionTimeMillisEstimate': 0,
                                                                               'works': 101, 'advanced': 101,
                                                                               'needTime': 0, 'needYield': 0,
                                                                               'saveState': 1, 'restoreState': 1,
                                                                               'isEOF': 0, 'invalidates': 0,
                                                                               'docsExamined': 101, 'alreadyHasObj': 0,
                                                                               'inputStage': {'stage': 'IXSCAN',
                                                                                              'nReturned': 101,
                                                                                              'executionTimeMillisEstimate': 0,
                                                                                              'works': 101,
                                                                                              'advanced': 101,
                                                                                              'needTime': 0,
                                                                                              'needYield': 0,
                                                                                              'saveState': 1,
                                                                                              'restoreState': 1,
                                                                                              'isEOF': 0,
                                                                                              'invalidates': 0,
                                                                                              'keyPattern': {
                                                                                                  'isOriginalTitle': 1},
                                                                                              'indexName': 'isOriginalTitle_1',
                                                                                              'isMultiKey': False,
                                                                                              'multiKeyPaths': {
                                                                                                  'isOriginalTitle': []},
                                                                                              'isUnique': False,
                                                                                              'isSparse': False,
                                                                                              'isPartial': False,
                                                                                              'indexVersion': 2,
                                                                                              'direction': 'forward',
                                                                                              'indexBounds': {
                                                                                                  'isOriginalTitle': [
                                                                                                      '[0, 0]']},
                                                                                              'keysExamined': 101,
                                                                                              'seeks': 1,
                                                                                              'dupsTested': 0,
                                                                                              'dupsDropped': 0,
                                                                                              'seenInvalidated': 0}}},
                                {'nReturned': 101, 'executionTimeMillisEstimate': 0, 'totalKeysExamined': 101,
                                 'totalDocsExamined': 101, 'executionStages': {'stage': 'FETCH', 'nReturned': 101,
                                                                               'executionTimeMillisEstimate': 0,
                                                                               'works': 101, 'advanced': 101,
                                                                               'needTime': 0, 'needYield': 0,
                                                                               'saveState': 25237,
                                                                               'restoreState': 25237, 'isEOF': 0,
                                                                               'invalidates': 0, 'docsExamined': 101,
                                                                               'alreadyHasObj': 0,
                                                                               'inputStage': {'stage': 'IXSCAN',
                                                                                              'nReturned': 101,
                                                                                              'executionTimeMillisEstimate': 0,
                                                                                              'works': 101,
                                                                                              'advanced': 101,
                                                                                              'needTime': 0,
                                                                                              'needYield': 0,
                                                                                              'saveState': 25237,
                                                                                              'restoreState': 25237,
                                                                                              'isEOF': 0,
                                                                                              'invalidates': 0,
                                                                                              'keyPattern': {
                                                                                                  'isOriginalTitle': -1},
                                                                                              'indexName': 'isOriginalTitle_-1',
                                                                                              'isMultiKey': False,
                                                                                              'multiKeyPaths': {
                                                                                                  'isOriginalTitle': []},
                                                                                              'isUnique': False,
                                                                                              'isSparse': False,
                                                                                              'isPartial': False,
                                                                                              'indexVersion': 2,
                                                                                              'direction': 'forward',
                                                                                              'indexBounds': {
                                                                                                  'isOriginalTitle': [
                                                                                                      '[0, 0]']},
                                                                                              'keysExamined': 101,
                                                                                              'seeks': 1,
                                                                                              'dupsTested': 0,
                                                                                              'dupsDropped': 0,
                                                                                              'seenInvalidated': 0}}}]},
         'serverInfo': {'host': 'kikyo.cl.cam.ac.uk', 'port': 27017, 'version': '4.0.9',
                        'gitVersion': 'fc525e2d9b0e4bceff5c2201457e564362909765'}, 'ok': 1.0},
        ################
        {'queryPlanner': {'plannerVersion': 1, 'namespace': 'imdb.imdb_all', 'indexFilterSet': False,
                          'parsedQuery': {'$and': [{'language': {'$eq': 'en'}}, {'types': {'$eq': 'festival'}}]},
                          'winningPlan': {'stage': 'FETCH',
                                          'inputStage': {'stage': 'IXSCAN', 'keyPattern': {'language': 1, 'types': -1},
                                                         'indexName': 'language_1_types_-1', 'isMultiKey': False,
                                                         'multiKeyPaths': {'language': [], 'types': []},
                                                         'isUnique': False, 'isSparse': False, 'isPartial': False,
                                                         'indexVersion': 2, 'direction': 'forward',
                                                         'indexBounds': {'language': ['[""en"", ""en""]'],
                                                                         'types': ['[""festival"", ""festival""]']}}},
                          'rejectedPlans': [{'stage': 'FETCH', 'filter': {'language': {'$eq': 'en'}},
                                             'inputStage': {'stage': 'IXSCAN',
                                                            'keyPattern': {'types': 1, 'primaryTitle': 1, 'title': -1},
                                                            'indexName': 'types_1_primaryTitle_1_title_-1',
                                                            'isMultiKey': False,
                                                            'multiKeyPaths': {'types': [], 'primaryTitle': [],
                                                                              'title': []}, 'isUnique': False,
                                                            'isSparse': False, 'isPartial': False, 'indexVersion': 2,
                                                            'direction': 'forward',
                                                            'indexBounds': {'types': ['[""festival"", ""festival""]'],
                                                                            'primaryTitle': ['[MinKey, MaxKey]'],
                                                                            'title': ['[MaxKey, MinKey]']}}}]},
         'executionStats': {'executionSuccess': True, 'nReturned': 2810, 'executionTimeMillis': 6,
                            'totalKeysExamined': 2810, 'totalDocsExamined': 2810,
                            'executionStages': {'stage': 'FETCH', 'nReturned': 2810, 'executionTimeMillisEstimate': 10,
                                                'works': 2811, 'advanced': 2810, 'needTime': 0, 'needYield': 0,
                                                'saveState': 23, 'restoreState': 23, 'isEOF': 1, 'invalidates': 0,
                                                'docsExamined': 2810, 'alreadyHasObj': 0,
                                                'inputStage': {'stage': 'IXSCAN', 'nReturned': 2810,
                                                               'executionTimeMillisEstimate': 10, 'works': 2811,
                                                               'advanced': 2810, 'needTime': 0, 'needYield': 0,
                                                               'saveState': 23, 'restoreState': 23, 'isEOF': 1,
                                                               'invalidates': 0,
                                                               'keyPattern': {'language': 1, 'types': -1},
                                                               'indexName': 'language_1_types_-1', 'isMultiKey': False,
                                                               'multiKeyPaths': {'language': [], 'types': []},
                                                               'isUnique': False, 'isSparse': False, 'isPartial': False,
                                                               'indexVersion': 2, 'direction': 'forward',
                                                               'indexBounds': {'language': ['[""en"", ""en""]'],
                                                                               'types': [
                                                                                   '[""festival"", ""festival""]']},
                                                               'keysExamined': 2810, 'seeks': 1, 'dupsTested': 0,
                                                               'dupsDropped': 0, 'seenInvalidated': 0}},
                            'allPlansExecution': [
                                {'nReturned': 101, 'executionTimeMillisEstimate': 0, 'totalKeysExamined': 101,
                                 'totalDocsExamined': 101, 'executionStages': {'stage': 'FETCH', 'nReturned': 101,
                                                                               'executionTimeMillisEstimate': 0,
                                                                               'works': 101, 'advanced': 101,
                                                                               'needTime': 0, 'needYield': 0,
                                                                               'saveState': 1, 'restoreState': 1,
                                                                               'isEOF': 0, 'invalidates': 0,
                                                                               'docsExamined': 101, 'alreadyHasObj': 0,
                                                                               'inputStage': {'stage': 'IXSCAN',
                                                                                              'nReturned': 101,
                                                                                              'executionTimeMillisEstimate': 0,
                                                                                              'works': 101,
                                                                                              'advanced': 101,
                                                                                              'needTime': 0,
                                                                                              'needYield': 0,
                                                                                              'saveState': 1,
                                                                                              'restoreState': 1,
                                                                                              'isEOF': 0,
                                                                                              'invalidates': 0,
                                                                                              'keyPattern': {
                                                                                                  'language': 1,
                                                                                                  'types': -1},
                                                                                              'indexName': 'language_1_types_-1',
                                                                                              'isMultiKey': False,
                                                                                              'multiKeyPaths': {
                                                                                                  'language': [],
                                                                                                  'types': []},
                                                                                              'isUnique': False,
                                                                                              'isSparse': False,
                                                                                              'isPartial': False,
                                                                                              'indexVersion': 2,
                                                                                              'direction': 'forward',
                                                                                              'indexBounds': {
                                                                                                  'language': [
                                                                                                      '[""en"", ""en""]'],
                                                                                                  'types': [
                                                                                                      '[""festival"", ""festival""]']},
                                                                                              'keysExamined': 101,
                                                                                              'seeks': 1,
                                                                                              'dupsTested': 0,
                                                                                              'dupsDropped': 0,
                                                                                              'seenInvalidated': 0}}},
                                {'nReturned': 4, 'executionTimeMillisEstimate': 0, 'totalKeysExamined': 101,
                                 'totalDocsExamined': 101,
                                 'executionStages': {'stage': 'FETCH', 'filter': {'language': {'$eq': 'en'}},
                                                     'nReturned': 4, 'executionTimeMillisEstimate': 0, 'works': 101,
                                                     'advanced': 4, 'needTime': 97, 'needYield': 0, 'saveState': 23,
                                                     'restoreState': 23, 'isEOF': 0, 'invalidates': 0,
                                                     'docsExamined': 101, 'alreadyHasObj': 0,
                                                     'inputStage': {'stage': 'IXSCAN', 'nReturned': 101,
                                                                    'executionTimeMillisEstimate': 0, 'works': 101,
                                                                    'advanced': 101, 'needTime': 0, 'needYield': 0,
                                                                    'saveState': 23, 'restoreState': 23, 'isEOF': 0,
                                                                    'invalidates': 0,
                                                                    'keyPattern': {'types': 1, 'primaryTitle': 1,
                                                                                   'title': -1},
                                                                    'indexName': 'types_1_primaryTitle_1_title_-1',
                                                                    'isMultiKey': False,
                                                                    'multiKeyPaths': {'types': [], 'primaryTitle': [],
                                                                                      'title': []}, 'isUnique': False,
                                                                    'isSparse': False, 'isPartial': False,
                                                                    'indexVersion': 2, 'direction': 'forward',
                                                                    'indexBounds': {
                                                                        'types': ['[""festival"", ""festival""]'],
                                                                        'primaryTitle': ['[MinKey, MaxKey]'],
                                                                        'title': ['[MaxKey, MinKey]']},
                                                                    'keysExamined': 101, 'seeks': 1, 'dupsTested': 0,
                                                                    'dupsDropped': 0, 'seenInvalidated': 0}}}]},
         'serverInfo': {'host': 'kikyo.cl.cam.ac.uk', 'port': 27017, 'version': '4.0.9',
                        'gitVersion': 'fc525e2d9b0e4bceff5c2201457e564362909765'}, 'ok': 1.0},
        ##################
        {'queryPlanner': {'plannerVersion': 1, 'namespace': 'imdb.imdb_all', 'indexFilterSet': False,
                          'parsedQuery': {'title': {'$eq': 'Around a Bathing Hut'}}, 'winningPlan': {'stage': 'FETCH',
                                                                                                     'inputStage': {
                                                                                                         'stage': 'IXSCAN',
                                                                                                         'keyPattern': {
                                                                                                             'title': 1},
                                                                                                         'indexName': 'title_1',
                                                                                                         'isMultiKey': False,
                                                                                                         'multiKeyPaths': {
                                                                                                             'title': []},
                                                                                                         'isUnique': False,
                                                                                                         'isSparse': False,
                                                                                                         'isPartial': False,
                                                                                                         'indexVersion': 2,
                                                                                                         'direction': 'forward',
                                                                                                         'indexBounds': {
                                                                                                             'title': [
                                                                                                                 '[""Around a Bathing Hut"", ""Around a Bathing Hut""]']}}},
                          'rejectedPlans': []},
         'executionStats': {'executionSuccess': True, 'nReturned': 1, 'executionTimeMillis': 0, 'totalKeysExamined': 1,
                            'totalDocsExamined': 1,
                            'executionStages': {'stage': 'FETCH', 'nReturned': 1, 'executionTimeMillisEstimate': 0,
                                                'works': 2, 'advanced': 1, 'needTime': 0, 'needYield': 0,
                                                'saveState': 0, 'restoreState': 0, 'isEOF': 1, 'invalidates': 0,
                                                'docsExamined': 1, 'alreadyHasObj': 0,
                                                'inputStage': {'stage': 'IXSCAN', 'nReturned': 1,
                                                               'executionTimeMillisEstimate': 0, 'works': 2,
                                                               'advanced': 1, 'needTime': 0, 'needYield': 0,
                                                               'saveState': 0, 'restoreState': 0, 'isEOF': 1,
                                                               'invalidates': 0, 'keyPattern': {'title': 1},
                                                               'indexName': 'title_1', 'isMultiKey': False,
                                                               'multiKeyPaths': {'title': []}, 'isUnique': False,
                                                               'isSparse': False, 'isPartial': False, 'indexVersion': 2,
                                                               'direction': 'forward', 'indexBounds': {'title': [
                                                        '[""Around a Bathing Hut"", ""Around a Bathing Hut""]']},
                                                               'keysExamined': 1, 'seeks': 1, 'dupsTested': 0,
                                                               'dupsDropped': 0, 'seenInvalidated': 0}},
                            'allPlansExecution': []},
         'serverInfo': {'host': 'kikyo.cl.cam.ac.uk', 'port': 27017, 'version': '4.0.9',
                        'gitVersion': 'fc525e2d9b0e4bceff5c2201457e564362909765'}, 'ok': 1.0},
        ##############
        {'queryPlanner': {'plannerVersion': 1, 'namespace': 'imdb.imdb_all', 'indexFilterSet': False,
                          'parsedQuery': {'$or': [{'language': {'$eq': '\\N'}}, {'startYear': {'$lt': 1964}}]},
                          'winningPlan': {'stage': 'SUBPLAN', 'inputStage': {'stage': 'FETCH',
                                                                             'inputStage': {'stage': 'OR',
                                                                                            'inputStages': [
                                                                                                {'stage': 'IXSCAN',
                                                                                                 'keyPattern': {
                                                                                                     'startYear': 1,
                                                                                                     'language': -1},
                                                                                                 'indexName': 'startYear_1_language_-1',
                                                                                                 'isMultiKey': False,
                                                                                                 'multiKeyPaths': {
                                                                                                     'startYear': [],
                                                                                                     'language': []},
                                                                                                 'isUnique': False,
                                                                                                 'isSparse': False,
                                                                                                 'isPartial': False,
                                                                                                 'indexVersion': 2,
                                                                                                 'direction': 'forward',
                                                                                                 'indexBounds': {
                                                                                                     'startYear': [
                                                                                                         '[-inf.0, 1964)'],
                                                                                                     'language': [
                                                                                                         '[MaxKey, MinKey]']}},
                                                                                                {'stage': 'IXSCAN',
                                                                                                 'keyPattern': {
                                                                                                     'language': 1,
                                                                                                     'types': -1},
                                                                                                 'indexName': 'language_1_types_-1',
                                                                                                 'isMultiKey': False,
                                                                                                 'multiKeyPaths': {
                                                                                                     'language': [],
                                                                                                     'types': []},
                                                                                                 'isUnique': False,
                                                                                                 'isSparse': False,
                                                                                                 'isPartial': False,
                                                                                                 'indexVersion': 2,
                                                                                                 'direction': 'forward',
                                                                                                 'indexBounds': {
                                                                                                     'language': [
                                                                                                         '[""\\N"", ""\\N""]'],
                                                                                                     'types': [
                                                                                                         '[MaxKey, MinKey]']}}]}}},
                          'rejectedPlans': []},
         'executionStats': {'executionSuccess': True, 'nReturned': 3680811, 'executionTimeMillis': 6335,
                            'totalKeysExamined': 3680811, 'totalDocsExamined': 3680811,
                            'executionStages': {'stage': 'SUBPLAN', 'nReturned': 3680811,
                                                'executionTimeMillisEstimate': 5532, 'works': 3680813,
                                                'advanced': 3680811, 'needTime': 1, 'needYield': 0, 'saveState': 28759,
                                                'restoreState': 28759, 'isEOF': 1, 'invalidates': 0,
                                                'inputStage': {'stage': 'FETCH', 'nReturned': 3680811,
                                                               'executionTimeMillisEstimate': 5462, 'works': 3680813,
                                                               'advanced': 3680811, 'needTime': 1, 'needYield': 0,
                                                               'saveState': 28759, 'restoreState': 28759, 'isEOF': 1,
                                                               'invalidates': 0, 'docsExamined': 3680811,
                                                               'alreadyHasObj': 0,
                                                               'inputStage': {'stage': 'OR', 'nReturned': 3680811,
                                                                              'executionTimeMillisEstimate': 2842,
                                                                              'works': 3680813, 'advanced': 3680811,
                                                                              'needTime': 1, 'needYield': 0,
                                                                              'saveState': 28759, 'restoreState': 28759,
                                                                              'isEOF': 1, 'invalidates': 0,
                                                                              'dupsTested': 3680811, 'dupsDropped': 0,
                                                                              'recordIdsForgotten': 0, 'inputStages': [
                                                                       {'stage': 'IXSCAN', 'nReturned': 346827,
                                                                        'executionTimeMillisEstimate': 150,
                                                                        'works': 346828, 'advanced': 346827,
                                                                        'needTime': 0, 'needYield': 0,
                                                                        'saveState': 28759, 'restoreState': 28759,
                                                                        'isEOF': 1, 'invalidates': 0,
                                                                        'keyPattern': {'startYear': 1, 'language': -1},
                                                                        'indexName': 'startYear_1_language_-1',
                                                                        'isMultiKey': False,
                                                                        'multiKeyPaths': {'startYear': [],
                                                                                          'language': []},
                                                                        'isUnique': False, 'isSparse': False,
                                                                        'isPartial': False, 'indexVersion': 2,
                                                                        'direction': 'forward',
                                                                        'indexBounds': {'startYear': ['[-inf.0, 1964)'],
                                                                                        'language': [
                                                                                            '[MaxKey, MinKey]']},
                                                                        'keysExamined': 346827, 'seeks': 1,
                                                                        'dupsTested': 0, 'dupsDropped': 0,
                                                                        'seenInvalidated': 0},
                                                                       {'stage': 'IXSCAN', 'nReturned': 3333984,
                                                                        'executionTimeMillisEstimate': 1770,
                                                                        'works': 3333985, 'advanced': 3333984,
                                                                        'needTime': 0, 'needYield': 0,
                                                                        'saveState': 28759, 'restoreState': 28759,
                                                                        'isEOF': 1, 'invalidates': 0,
                                                                        'keyPattern': {'language': 1, 'types': -1},
                                                                        'indexName': 'language_1_types_-1',
                                                                        'isMultiKey': False,
                                                                        'multiKeyPaths': {'language': [], 'types': []},
                                                                        'isUnique': False, 'isSparse': False,
                                                                        'isPartial': False, 'indexVersion': 2,
                                                                        'direction': 'forward', 'indexBounds': {
                                                                           'language': ['[""\\N"", ""\\N""]'],
                                                                           'types': ['[MaxKey, MinKey]']},
                                                                        'keysExamined': 3333984, 'seeks': 1,
                                                                        'dupsTested': 0, 'dupsDropped': 0,
                                                                        'seenInvalidated': 0}]}}},
                            'allPlansExecution': []},
         'serverInfo': {'host': 'kikyo.cl.cam.ac.uk', 'port': 27017, 'version': '4.0.9',
                        'gitVersion': 'fc525e2d9b0e4bceff5c2201457e564362909765'}, 'ok': 1.0},
        ########################
        {'queryPlanner': {'plannerVersion': 1, 'namespace': 'imdb.imdb_all', 'indexFilterSet': False,
                          'parsedQuery': {'$or': [{'isAdult': {'$eq': 1}}, {'runtimeMinutes': {'$eq': 120}}]},
                          'winningPlan': {'stage': 'SUBPLAN', 'inputStage': {'stage': 'COLLSCAN', 'filter': {
                              '$or': [{'isAdult': {'$eq': 1}}, {'runtimeMinutes': {'$eq': 120}}]},
                                                                             'direction': 'forward'}},
                          'rejectedPlans': []},
         'executionStats': {'executionSuccess': True, 'nReturned': 162867, 'executionTimeMillis': 4631,
                            'totalKeysExamined': 0, 'totalDocsExamined': 9392574,
                            'executionStages': {'stage': 'SUBPLAN', 'nReturned': 162867,
                                                'executionTimeMillisEstimate': 4190, 'works': 9392576,
                                                'advanced': 162867, 'needTime': 9229708, 'needYield': 0,
                                                'saveState': 73379, 'restoreState': 73379, 'isEOF': 1, 'invalidates': 0,
                                                'inputStage': {'stage': 'COLLSCAN', 'filter': {
                                                    '$or': [{'isAdult': {'$eq': 1}}, {'runtimeMinutes': {'$eq': 120}}]},
                                                               'nReturned': 162867, 'executionTimeMillisEstimate': 4090,
                                                               'works': 9392576, 'advanced': 162867,
                                                               'needTime': 9229708, 'needYield': 0, 'saveState': 73379,
                                                               'restoreState': 73379, 'isEOF': 1, 'invalidates': 0,
                                                               'direction': 'forward', 'docsExamined': 9392574}},
                            'allPlansExecution': []},
         'serverInfo': {'host': 'kikyo.cl.cam.ac.uk', 'port': 27017, 'version': '4.0.9',
                        'gitVersion': 'fc525e2d9b0e4bceff5c2201457e564362909765'}, 'ok': 1.0}

    ]

    def test_counting(self):
        """
        Tests counting of index stats from explain.
        """
        index_used = {}
        for stats in self.all_stats:
            # Consider winning plan.
            if "cursor" in stats:
                print("Cursor is in stats = {}, type= {}".format(stats["cursor"], type(stats["cursor"])))
                # 'cursor': 'BtreeCursor numVotes_1_titleType_-1_titleType_1
                if isinstance(stats["cursor"], str):
                    cursor_components = stats["cursor"].split(" ")
                    if cursor_components[0] == "BtreeCursor":
                        print("BtreeCursor counted.")
                        index_name = cursor_components[1]
                        # Was this index name used?
                        index_used[index_name][0] += 1
                        # Create a list of the lengths of queries using this for intersection.
                        index_used[index_name][1].append(str(3))
                    else:
                        print("Cursor {} not counted.".format(cursor_components[0]))
                # Plan-response.
            elif "queryPlanner" in stats:
                plan_response = stats["queryPlanner"]
                assert "winningPlan" in plan_response, "Key `winningPlan` not in plan response"
                winning_plan = plan_response["winningPlan"]
                # Winning plan has a recursive structure of different stages. We are looking for
                # an index name used in a stage IXSCAN.
                if "inputStage" in winning_plan:
                    top_level_stage = winning_plan["inputStage"]
                    if top_level_stage == "SUBPLAN":
                        # Skip to subplan.
                        top_level_stage = winning_plan["inputStage"]
                    if top_level_stage["stage"] == "IXSCAN":
                        self.check_stage(index_used, top_level_stage)
                    # OR can fetch multiple indices.
                    elif top_level_stage["stage"] == "FETCH":
                        if top_level_stage["inputStage"]["stage"] == "OR":
                            stages = top_level_stage["inputStage"]["inputStages"]
                            for stage in stages:
                                if stage["stage"] == "IXSCAN":
                                    self.check_stage(index_used, stage)
                        elif top_level_stage["inputStage"]["stage"] == "IXSCAN":
                            self.check_stage(index_used, top_level_stage["inputStage"]["stage"])
                    elif top_level_stage["stage"] == "COLLSCAN":
                        print("COLLSCAN")
            else:
                print("No cursor found, dumping raw stats: {} ".format(stats))
        print(index_used)
        self.assertEqual(index_used, self.expected)

    def check_stage(self, index_used, stage):
        index_name = stage["indexName"]
        if index_name not in index_used:
            index_used[index_name] = 0
        index_used[index_name] += 1
