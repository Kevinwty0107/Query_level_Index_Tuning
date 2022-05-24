from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import json
import gflags

from lift.case_studies.mongodb.gpy_opt_mongodb import GPyOptMongoDB
from lift.case_studies.mongodb import mongo_schemas, mongo_model_generators, MongoDataSource
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

    # We may not always have separate test data.
    if FLAGS.test_label == '':
        test_dir = None
    else:
        test_dir = FLAGS.trace_dir

    # Build schema, models.
    schema = mongo_schemas[experiment_config['model']](
            schema_config=schema_config,
            experiment_config=experiment_config
        )
    system_model = MongoDBSystemEnvironment(
        experiment_config=experiment_config,
        host=FLAGS.host
    )

    sample_values = experiment_config.get('sample_values', False)
    test_queries = load_queries(system_model=system_model, query_dir=test_dir,
                                label=FLAGS.test_label, sample_query_values=sample_values)

    # Generate queries.
    gpyopt = GPyOptMongoDB(
        queries=test_queries,
        experiment_config=experiment_config,
        schema=schema,
        result_dir=FLAGS.result_dir,
        system_model=system_model
    )

    # Runs GPyOpt optimization, exports results.
    gpyopt.run(FLAGS.result_label, experiment_config["training_steps"])


if __name__ == '__main__':
    main(sys.argv)
