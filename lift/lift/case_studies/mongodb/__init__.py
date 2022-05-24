
from lift.case_studies.mongodb.deprecated.combinatorial_data_loader import CombinatorialDataSource
from lift.case_studies.mongodb.deprecated.combinatorial_expiration_model import CombinatorialExpirationModel
from lift.case_studies.mongodb.combinatorial_converter import CombinatorialConverter
from lift.case_studies.mongodb.combinatorial_schema import CombinatorialSchema
from lift.case_studies.mongodb.field_position_converter import FieldPositionConverter
from lift.case_studies.mongodb.field_position_schema import FieldPositionSchema
from lift.case_studies.mongodb.deprecated.mongo_data_source import MongoDataSource
from lift.case_studies.mongodb.deprecated.sequence_model_generator import SequenceConverter
from lift.case_studies.mongodb.deprecated.sequence_schema import SequenceSchema
from lift.case_studies.mongodb.mongo_demonstration_rules import MongoFullIndexing, MongoNegativeRule, MongoExpert, \
    MongoNoop, MongoSingleColumn, MongoPrefixRule

mongo_model_generators = dict(
    combinatorial=CombinatorialConverter,
    field_position=FieldPositionConverter,
    sequence=SequenceConverter
)

mongo_schemas = dict(
    combinatorial=CombinatorialSchema,
    field_position=FieldPositionSchema,
    sequence=SequenceSchema
)

mongo_system_models = dict(
    combinatorial=CombinatorialExpirationModel
)


mongo_demo_rules = {
    "full_indexing": MongoFullIndexing,
    "expert_indexing": MongoExpert,
    "negative": MongoNegativeRule,
    "noop": MongoNoop,
    "single_column": MongoSingleColumn,
    "prefix_heuristic": MongoPrefixRule
}