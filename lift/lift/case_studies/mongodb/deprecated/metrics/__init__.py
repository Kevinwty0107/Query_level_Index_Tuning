from lift.case_studies.mongodb.deprecated.metrics.mongo_metrics_helper import MongoMetricsHelper

metrics_helpers = dict(
    mongodb=MongoMetricsHelper
)

__all__ = [
    'MongoMetricsHelper'
]