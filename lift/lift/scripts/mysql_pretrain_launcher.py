from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import logging
import json
import gflags
import numpy as np
from lift.case_studies.mysql.mysql_pretrain_controller import MySQLPretrainController

FLAGS = gflags.FLAGS

gflags.DEFINE_string('agent', '', 'Agent configuration .json')
gflags.DEFINE_string('network', None, 'Network configuration json')
gflags.DEFINE_string('schema', '', 'Schema configuration json')
gflags.DEFINE_string('experiment_config', '', 'General experiment configuration json')
gflags.DEFINE_string('result_dir', '', 'Path to result directory, end with /')
gflags.DEFINE_string('model', '', 'Path to model checkpoint to import model')

gflags.DEFINE_boolean('load', False, 'Whether to load a trained model')
gflags.DEFINE_string('train_label', '', 'Name of training trace.')
gflags.DEFINE_string('test_label', '', 'Name of test trace.')
gflags.DEFINE_string('trace_dir', '', 'Path to (weakly) supervised train data.')

gflags.DEFINE_boolean('blackbox_optimization_mode', False, 'If true, training and test data are the same as we are searching'
                       'a single solution instead of a model which can generalise.')
gflags.DEFINE_integer('seed', 0, 'Numpy random seed')


logging.basicConfig()


def main(argv):
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    logging_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                          datefmt='%y-%m-%d %H:%M:%S')
    root_logger = logging.getLogger('')
    print_logging_handler = logging.StreamHandler(stream=sys.stdout)
    print_logging_handler.setFormatter(logging_formatter)
    print_logging_handler.setLevel(level=logging.INFO)
    root_logger.setLevel(level=logging.DEBUG)
    root_logger.addHandler(print_logging_handler)

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

    # Use provided seed or sample one.
    if FLAGS.seed == 0:
        seed = np.random.randint(100000)
        print("Starting with randomly sampled seed:", seed)
    else:
        seed = FLAGS.seed
        print("Starting with provided seed:", seed)
    # Store this seed so we can reproduce this workload sequence again.
    np.random.seed(seed=seed)
    np.savetxt(FLAGS.result_dir + '/pretrain_numpy_random_seed.txt', np.asarray([seed]), delimiter=',')
    controller = MySQLPretrainController(
        agent_config=agent_config,
        network_config=network_config,
        experiment_config=experiment_config,
        schema_config=schema_config,
        result_dir=FLAGS.result_dir,
        model_path=FLAGS.model,
        load_model=FLAGS.load,
        training_dir=FLAGS.trace_dir,
        test_dir=test_dir,  # Both in same dir,
        blackbox_mode=FLAGS.blackbox_optimization_mode
    )
    controller.run()

    # Export the episode used for demos as both training and test data in blackbox mode.
    if FLAGS.blackbox_optimization_mode is True:
        controller.data_source.export_data(
            data=controller.queries,
            data_dir=FLAGS.result_dir,
            label="mysql_train")
        controller.data_source.export_data(
            data=controller.queries,
            data_dir=FLAGS.result_dir,
            label="mysql_test")


if __name__ == '__main__':
    main(sys.argv)