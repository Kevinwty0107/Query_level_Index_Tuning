from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import json
import gflags

from lift.case_studies.mysql.open_tuner_mysql import OpenTunerMySQL
from lift.case_studies.mysql.mysql_converter import MySQLConverter
from lift.case_studies.mysql.mysql_data_source import MySQLDataSource
from lift.case_studies.mysql.mysql_schema import MySQLSchema
from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment

FLAGS = gflags.FLAGS

gflags.DEFINE_string('schema', '', 'Schema configuration json')
gflags.DEFINE_string('experiment_config', '', 'General experiment configuration json')
gflags.DEFINE_string('result_dir', '', 'Path to result directory, end with /')

gflags.DEFINE_string('train_label', '', 'Name of training trace.')
gflags.DEFINE_string('test_label', '', 'Name of test trace.')
gflags.DEFINE_string('result_label', '', 'Result label.')


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
    schema = MySQLSchema(
        schema_config=schema_config
        )

    system_model = MySQLSystemEnvironment(
        all_tables=schema_config["tables"]
    )
    converter = MySQLConverter(
        experiment_config=experiment_config,
        schema=schema
    )
    data_source = MySQLDataSource(
        converter=converter,
        schema=schema
    )
    train_queries = data_source.load_data(FLAGS.result_dir, label=FLAGS.train_label)
    test_queries = data_source.load_data(FLAGS.result_dir, label=FLAGS.test_label)

    # Generate queries.
    opentuner = OpenTunerMySQL(
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
