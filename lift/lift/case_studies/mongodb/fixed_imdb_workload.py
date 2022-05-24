from lift.case_studies.common.query_workload import QueryWorkload
from lift.case_studies.mongodb.imdb_util import imdb_sampling_fns
from lift.case_studies.mongodb.mongodb_query import MongoDBQuery




class FixedIMDBWorkload(QueryWorkload):
    """
    Fixed query set: Demo, train and test are the same.

    """
    def __init__(self):
        self.queries = []
        self._define_queries()

    def _define_queries(self):
        """
        Defines a fixed set of application queries.
        """

        query_filter = {"$and": [lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}},
                                 lambda: {"startYear": {"$gt": imdb_sampling_fns["startYear"]()}}]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q1',
            query_dict=query_dict,
            tokens=["$and", "titleType_1", "$eq", "startYear_1", "$gt"],
            index_columns=[("titleType", 1), ("startYear", 1)],
            query_columns=["titleType", "startYear"]
        ))

        query_filter = {"$and": [
            lambda: {"region": {"$eq": imdb_sampling_fns["region"]()}},
            lambda: {"isOriginalTitle": {"$eq": imdb_sampling_fns["isOriginalTitle"]()}}]
        }
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q2',
            query_dict=query_dict,
            tokens=["$and", "region_1", "$eq", "isOriginalTitle_1", "$eq"],
            index_columns= [("region", 1), ("isOriginalTitle", 1)],
            query_columns=["region", "isOriginalTitle"]
        ))

        query_filter = {"$and": [
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}},
            lambda: {"runtimeMinutes": {"$gt": imdb_sampling_fns["runtimeMinutes"]()}},
            lambda: {"isAdult": {"$eq": imdb_sampling_fns["isAdult"]()}},
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[('runtimeMinutes', -1)])
        self.queries.append(MongoDBQuery(
            name='q3',
            query_dict=query_dict,
            tokens=["$and", "titleType_1","$eq", "runtimeMinutes_-1", "$gt", "isAdult_1", "$eq"],
            index_columns=[("titleType", 1), ("runtimeMinutes", 1)],
            query_columns=["titleType", "runtimeMinutes", "isAdult"]
        ))

        query_filter = {"$and": [
            lambda: {"genres": {"$eq": imdb_sampling_fns["genres"]()}},
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}},
            lambda: {"isAdult": {"$eq": imdb_sampling_fns["isAdult"]()}}]
        }
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q4',
            query_dict=query_dict,
            tokens=["$and", "genres_1", "$eq", "titleType_1", "$eq", "isAdult_1", "$eq"],
            index_columns=[("genres", 1), ("titleType", 1)],
            query_columns=["genres", "titleType", "isAdult"]
        ))

        query_filter = {"$and": [
            lambda: {"startYear": {"$gt": imdb_sampling_fns["startYear"]()}},
            lambda: {"endYear": {"$lt": imdb_sampling_fns["endYear"]()}},
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}}]
        }
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[('startYear', -1), ('endYear', -1)])
        self.queries.append(MongoDBQuery(
            name='q5',
            query_dict=query_dict,
            tokens=["$and", "startYear_-1", "$gt", "endYear_-1", "$lt", "titleType_1", "$eq"],
            index_columns=[("startYear", 1), ("endYear", 1), ("titleType", 1)],
            query_columns=["startYear", "endYear", "titleType"]
        ))

        query_filter = {"$or": [
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}},
            lambda: {"runtimeMinutes": {"$lt": imdb_sampling_fns["runtimeMinutes"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q6',
            query_dict=query_dict,
            tokens=["$or", "titleType_1", "$eq","runtimeMinutes_1", "$lt"],
            index_columns=[],
            query_columns=["titleType", "runtimeMinutes"]
        ))

        query_filter = {"$and": [
            lambda: {"genres": {"$eq": imdb_sampling_fns["genres"]()}},
            lambda: {"runtimeMinutes": {"$gt": imdb_sampling_fns["runtimeMinutes"]()}},
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[('runtimeMinutes', -1)])
        self.queries.append(MongoDBQuery(
            name='q7',
            query_dict=query_dict,
            tokens=["$and", "genres_1", "$eq", "runtimeMinutes_-1", "$gt"],
            index_columns=[("genres", 1), ("runtimeMinutes", -1)],
            query_columns=["genres", "runtimeMinutes"]
        ))

        query_filter = {"$or": [
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}},
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}},
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q8',
            query_dict=query_dict,
            tokens=["$or", "titleType_1", "$eq", "titleType_1", "$eq"],
            index_columns=[],
            query_columns=["titleType"]
        ))

        query_filter = {"$and": [
            lambda: {"types": {"$eq": imdb_sampling_fns["types"]()}},
            lambda: {"region": {"$eq": imdb_sampling_fns["region"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q9',
            query_dict=query_dict,
            tokens=["$and", "types_1", "$eq", "region_1","$eq"],
            index_columns=[("types", 1), ("region", 1)],
            query_columns=["types", "region"]
        ))

        query_filter = {"and": [
           lambda: {"title": {"$eq": imdb_sampling_fns["title"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='limit', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q10',
            query_dict=query_dict,
            tokens=["title_1", "$eq"],
            index_columns=[("title", 1)],
            query_columns=["title"]
        ))

        query_filter = {"$and": [
            lambda: {"startYear": {"$lt": imdb_sampling_fns["startYear"]()}},
            lambda: {"genres": {"$eq": imdb_sampling_fns["genres"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[("start_year", -1)])
        self.queries.append(MongoDBQuery(
            name='q11',
            query_dict=query_dict,
            tokens=["$and", "startYear_-1","$lt","genres_1","$eq"],
            index_columns=[("startYear", -1), ("genres", 1)],
            query_columns=["startYear", "genres"]
        ))

        query_filter = {"$or": [
            lambda: {"types": {"$eq": imdb_sampling_fns["types"]()}},
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q12',
            query_dict=query_dict,
            tokens=["$or", "types_1", "$eq","titleType_1", "$eq"],
            index_columns=[("types", 1)],
            query_columns=["types", "titleType"]
        ))

        query_filter = {"$and": [
            lambda: {"startYear": {"$gt": imdb_sampling_fns["startYear"]()}},
            lambda: {"endYear": {"$lt": imdb_sampling_fns["endYear"]()}},
            lambda: {"genres": {"$eq": imdb_sampling_fns["genres"]()}},
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[("endYear", 1)])
        self.queries.append(MongoDBQuery(
            name='q13',
            query_dict=query_dict,
            tokens=["$and", "startYear_1", "$gt", "endYear_1", "$lt", "genres_1", "$eq"],
            # Covered by q5
            index_columns=[],
            query_columns=["startYear", "endYear", "genres"]
        ))

        query_filter = {"$and": [
            lambda: {"averageRating": {"$gt": imdb_sampling_fns["averageRating"]()}},
            lambda: {"numVotes": {"$gt": imdb_sampling_fns["numVotes"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q14',
            query_dict=query_dict,
            tokens=["$and", "averageRating_1", "$gt", "numVotes_1", "$gt"],
            index_columns=[("averageRating", 1), ("numVotes", 1)],
            query_columns=["averageRating", "numVotes"]
        ))

        query_filter = {"$and": [
            lambda: {"averageRating": {"eq": imdb_sampling_fns["averageRating"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q15',
            tokens=["averageRating_1", "$eq"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["averageRating"]
        ))

        query_filter = {"$and": [
            lambda: {"startYear": {"$lt": imdb_sampling_fns["startYear"]()}},
            lambda: {"genres": {"$eq": imdb_sampling_fns["genres"]()}},
            lambda: {"runtimeMinutes": {"$gt": imdb_sampling_fns["runtimeMinutes"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[("start_year", -1)])
        self.queries.append(MongoDBQuery(
            name='q16',
            tokens=["$and", "startYear_-1", "$lt", "genres_1", "$eq", "runtimeMinutes_1", "$gt"],
            index_columns=[("startYear", -1), ("genres", 1), ("runtimeMinutes", 1)],
            query_dict=query_dict,
            query_columns=["startYear", "genres","runtimeMinutes"]
        ))

        query_filter = {"$and": [
            lambda: {"isOriginalTitle": {"eq": imdb_sampling_fns["isOriginalTitle"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q17',
            tokens=["isOriginalTitle_1","$eq"],
            index_columns=[("isOriginalTitle", -1)],
            query_dict=query_dict,
            query_columns=["isOriginalTitle"]
        ))

        query_filter = {"$and": [
            lambda: {"startYear": {"$gt": imdb_sampling_fns["startYear"]()}},
            lambda: {"endYear": {"$lt": imdb_sampling_fns["endYear"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q18',
            tokens=["$and", "startYear_1", "$gt", "endYear_1", "$lt"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["startYear", "endYear"]
        ))

        query_filter = {"$and": [
            lambda: {"numVotes": {"$gt": imdb_sampling_fns["numVotes"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[("numVotes", -1)])
        self.queries.append(MongoDBQuery(
            name='q19',
            tokens=["numVotes_-1", "$gt"],
            index_columns=[("numVotes", -1)],
            query_dict=query_dict,
            query_columns=["numVotes"]
        ))
        query_filter = {"$and": [
            lambda: {"titleType": {"$eq": imdb_sampling_fns["titleType"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q20',
            tokens=["titleType_1", "$eq"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["titleType"]
        ))

        query_filter = {"$and": [
            lambda: {"numVotes": {"$gt": imdb_sampling_fns["numVotes"]()}},
            lambda: {"language": {"$eq": imdb_sampling_fns["language"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='count', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q21',
            tokens=["$and", "numVotes_1", "$gt", "language_1", "$eq"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["numVotes", "language"]
        ))

        query_filter = {"$and": [
            lambda: {"language": {"$eq": imdb_sampling_fns["language"]()}},
            lambda: {"primaryTitle": {"$eq": imdb_sampling_fns["primaryTitle"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='limit', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q22',
            tokens=["$and", "language_1", "$eq", "primaryTitle_1", "$eq"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["language", "primaryTitle"]
        ))

        query_filter = {"$or": [
            lambda: {"ordering": {"$eq": imdb_sampling_fns["ordering"]()}},
            lambda: {"ordering": {"$eq": imdb_sampling_fns["ordering"]()}}
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='limit', sort_order=[])
        self.queries.append(MongoDBQuery(
            name='q23',
            tokens=["$or", "ordering_1", "$eq", "ordering_1", "$eq"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["ordering"]
        ))

        query_filter = {"$or": [
            lambda: {"runtimeMinutes": {"$gt": imdb_sampling_fns["runtimeMinutes"]()}},
            lambda: {"numVotes": {"$gt": imdb_sampling_fns["numVotes"]()}},
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[("numVotes", -1)])
        self.queries.append(MongoDBQuery(
            name='q24',
            tokens=["$or", "runtimeMinutes_1", "$gt", "numVotes_-1", "$gt"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["runtimeMinutes", "numVotes"]
        ))

        query_filter = {"$or": [
            lambda: {"startYear": {"$gt": imdb_sampling_fns["startYear"]()}},
            lambda: {"runtimeMinutes": {"$lt": imdb_sampling_fns["runtimeMinutes"]()}},
        ]}
        query_dict = dict(query_filter=query_filter, aggregation='sort', sort_order=[("startYear", -1)])
        self.queries.append(MongoDBQuery(
            name='q25',
            tokens=["$or", "startYear_-1", "$gt", "numVotes_1", "$lt"],
            index_columns=[],
            query_dict=query_dict,
            query_columns=["startYear", "runtimeMinutes"]
        ))

    def define_demo_queries(self, num_queries=1):
        return self.queries

    def define_train_queries(self, num_queries=1):
        return self.queries

    def define_test_queries(self, num_queries=1):
        return self.queries
