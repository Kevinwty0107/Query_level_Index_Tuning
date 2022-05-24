from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import logging
import json
import gflags

logging_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                      datefmt='%y-%m-%d %H:%M:%S')
root_logger = logging.getLogger('')
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(level=logging.DEBUG)
print_logging_handler = logging.StreamHandler(stream=sys.stdout)
print_logging_handler.setFormatter(logging_formatter)
print_logging_handler.setLevel(level=logging.DEBUG)
root_logger.setLevel(level=logging.DEBUG)
root_logger.addHandler(print_logging_handler)

# Tensorflow import happens here ->setup logging before
from lift.case_studies.mongodb.deprecated.mongo_system_controller_deprecated import MongoSystemController

FLAGS = gflags.FLAGS

gflags.DEFINE_string('host', 'hasu', 'MongoDB host')
gflags.DEFINE_string('agent', '', 'Agent configuration .json')
gflags.DEFINE_string('network', None, 'Network configuration json')
gflags.DEFINE_string('schema', '', 'Schema configuration json')
gflags.DEFINE_string('experiment_config', '', 'General experiment configuration json')
gflags.DEFINE_string('result_dir', '', 'Path to result directory, end with /')

gflags.DEFINE_string('model_store_path', '', 'Path to load model checkpoint')
gflags.DEFINE_string('model_load_path', '', 'Path to store model checkpoint')
gflags.DEFINE_boolean('load', False, 'Whether to load a trained model')
gflags.DEFINE_boolean('store', False, 'Whether to store the final model')

gflags.DEFINE_string('demo_data_label', '', 'Label for demo data to be used in continued online training.')
gflags.DEFINE_string('train_label', '', 'Name of training trace.')
gflags.DEFINE_string('test_label', '', 'Name of test trace.')

gflags.DEFINE_string('trace_dir', '', 'Path to trace data for training.')

gflags.DEFINE_string('single_model_label', '', 'Label of single model loaded')


logging.basicConfig()


def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    with open(FLAGS.agent, 'r') as fp:
        agent_config = json.load(fp=fp)
    with open(FLAGS.experiment_config, 'r') as fp:
        experiment_config = json.load(fp=fp)
    with open(FLAGS.schema, 'r') as fp:
        schema_config = json.load(fp=fp)

    if FLAGS.network:
        with open(FLAGS.network, 'r') as fp:
            network_config = json.load(fp=fp)
    else:
        network_config = None
    # We may not always have separate test data.
    if FLAGS.test_label == '':
        test_dir = None
    else:
        test_dir = FLAGS.trace_dir

    controller = MongoSystemController(
        agent_config=agent_config,
        network_config=network_config,
        experiment_config=experiment_config,
        schema_config=schema_config,
        host=FLAGS.host,
        result_dir=FLAGS.result_dir,
        demo_data_label=FLAGS.demo_data_label,
        train_label=FLAGS.train_label,
        test_label=FLAGS.test_label,
        model_store_path=FLAGS.model_store_path,
        model_load_path=FLAGS.model_load_path,
        store_model=FLAGS.store,
        load_model=FLAGS.load,
        training_dir=FLAGS.trace_dir,
        test_dir=test_dir
    )
    controller.evaluate_tf_model(FLAGS.model_load_path, FLAGS.single_model_label)


if __name__ == '__main__':
    main(sys.argv)