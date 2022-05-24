from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import json
import gflags

from lift.case_studies.mongodb.fixed_imdb_workload import FixedIMDBWorkload
from lift.case_studies.mongodb.mongodb_data_source import MongoDBDataSource
from lift.case_studies.mongodb.open_tuner_mongodb import OpenTunerMongoDB
from lift.case_studies.mongodb import mongo_model_generators, MongoDataSource, FieldPositionSchema
from lift.case_studies.mongodb.mongodb_system_environment import MongoDBSystemEnvironment

FLAGS = gflags.FLAGS

gflags.DEFINE_string('host', 'hasu', 'MongoDB host')
gflags.DEFINE_string('schema', '', 'Schema configuration json')
gflags.DEFINE_string('experiment_config', '', 'General experiment configuration json')
gflags.DEFINE_string('result_dir', '', 'Path to result directory, end with /')

gflags.DEFINE_string('train_label', '', 'Name of training trace.')
gflags.DEFINE_string('test_label', '', 'Name of test trace.')
gflags.DEFINE_string('result_label', '', 'Result label.')

gflags.DEFINE_string('trace_dir', '', 'Path to trace data for training.')
gflags.DEFINE_boolean('fixed_workload', False, 'If true, use fid workload')


def load_queries(system_model, query_dir, label, sample_query_values):
    path = str(query_dir) + '/' + str(label) + 'queries.json'
    train_queries = MongoDataSource.load_query_dicts(query_path=path)

    return system_model.make_executable(
        queries=train_queries,
        sample_values=sample_query_values
    )


def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    with open(FLAGS.experiment_config, 'r') as fp:
        experiment_config = json.load(fp=fp)
    with open(FLAGS.schema, 'r') as fp:
        schema_config = json.load(fp=fp)

    # Build schema, models.
    schema = FieldPositionSchema(
        schema_config=schema_config,
        schema_spec=experiment_config["schema_spec"]
    )
    system_model = MongoDBSystemEnvironment(
        experiment_config=experiment_config,
        host=FLAGS.host
    )

    data_source = MongoDBDataSource(
        converter=None,
        schema=schema
    )

    if FLAGS.fixed_workload is True:
        workload = FixedIMDBWorkload()
        train_queries = workload.define_train_queries()
        test_queries = workload.define_test_queries()
    else:
        train_queries = data_source.load_data(FLAGS.result_dir, label=FLAGS.train_label)
        test_queries = data_source.load_data(FLAGS.result_dir, label=FLAGS.test_label)

    # Generate queries.
    opentuner = OpenTunerMongoDB(
        train_queries=train_queries,
        test_queries=test_queries,
        experiment_config=experiment_config,
        schema=schema,
        result_dir=FLAGS.result_dir,
        system_model=system_model
    )

    # Runs OpenTuner optimization, exports results.
    opentuner.run(FLAGS.result_label, experiment_config["open_tuner_training_steps"])


if __name__ == '__main__':
    main(sys.argv)
