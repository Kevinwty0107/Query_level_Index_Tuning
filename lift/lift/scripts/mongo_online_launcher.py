from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import sys
import logging
import json
import gflags

from lift.case_studies.mongodb.fixed_imdb_workload import FixedIMDBWorkload
from lift.case_studies.mongodb.imdb_synthetic_workload import IMDBSyntheticWorkload
from lift.case_studies.mongodb.mongo_demonstration_rules import MongoFullIndexing

logging_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                      datefmt='%y-%m-%d %H:%M:%S')
root_logger = logging.getLogger('')
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(level=logging.INFO)
print_logging_handler = logging.StreamHandler(stream=sys.stdout)
print_logging_handler.setFormatter(logging_formatter)
print_logging_handler.setLevel(level=logging.INFO)
root_logger.setLevel(level=logging.INFO)
root_logger.addHandler(print_logging_handler)

# Tensorflow import happens here ->setup logging before
from lift.case_studies.mongodb.mongo_system_controller import MongoSystemController
import tensorflow as tf

FLAGS = gflags.FLAGS

gflags.DEFINE_string('host', 'hasu', 'mongo host')
gflags.DEFINE_string('agent', '', 'Agent configuration .json')
gflags.DEFINE_string('network', '', 'Network configuration json')
gflags.DEFINE_string('schema', '', 'Schema configuration json')
gflags.DEFINE_string('experiment_config', '', 'General experiment configuration json')
gflags.DEFINE_string('result_dir', '', 'Path to result directory, end with /')

gflags.DEFINE_string('model_store_path', None, 'Path to load model checkpoint')
gflags.DEFINE_string('model_load_path', None, 'Path to store model checkpoint')

gflags.DEFINE_boolean('fixed_workload', False, 'If true, use fid workload')
gflags.DEFINE_boolean('generate_workload', False, 'If true, generate workload.')
gflags.DEFINE_boolean('restore_workload', False, 'If true, generate workload.')

# Eval flags;
gflags.DEFINE_boolean('train_online_only', False, 'Train without loading pretrained model.')
gflags.DEFINE_boolean('train_pretrain_online', False, 'Train with loading pretrained model.')
gflags.DEFINE_boolean('evaluate_default', False, 'Evaluate without indexing.')
gflags.DEFINE_boolean('evaluate_pretrain', False, 'Evaluate pretrained model.')
gflags.DEFINE_boolean('evaluate_demo_rule', False, 'Evaluate demo rule used for pretraining.')

gflags.DEFINE_boolean('blackbox_optimization_mode', False, 'If true, training and test data are the same as we are searching'
                                                  'a single solution instead of a model which can generalise.')


gflags.DEFINE_string('demo_data_label', '', 'Label for demo data to be used in continued online training.')
gflags.DEFINE_string('train_label', '', 'Name of training trace.')
gflags.DEFINE_string('test_label', '', 'Name of test trace.')
gflags.DEFINE_string('trace_dir', '', 'Path to trace data for training.')
gflags.DEFINE_integer('seed', 0, 'Numpy random seed')
gflags.DEFINE_integer('tf_seed', 0, 'TF random seed')


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

    # Use provided seed or sample one.
    if FLAGS.seed == 0:
        seed = np.random.randint(100000)
        print("Starting with randomly sampled seed:", seed)
    else:
        seed = FLAGS.seed

    if FLAGS.tf_seed == 0:
        tf_seed = np.random.randint(100000)
    else:
        tf_seed = FLAGS.tf_seed

    print("Starting with provided seed:", seed)
    tf.random.set_random_seed(tf_seed)
    # Store this seed so we can reproduce this workload sequence again.
    np.random.seed(seed=seed)

    if FLAGS.train_online_only:
        mode = 'online_only'
    else:
        mode = 'pretrain_online'

    np.savetxt(FLAGS.result_dir + '/{}_numpy_random_seed.txt'.format(mode), np.asarray([seed]), delimiter=',')
    np.savetxt(FLAGS.result_dir + '/{}_tf_random_seed.txt'.format(mode), np.asarray([tf_seed]), delimiter=',')

    controller = MongoSystemController(
        host=FLAGS.host,
        agent_config=agent_config,
        network_config=network_config,
        experiment_config=experiment_config,
        schema_config=schema_config,
        result_dir=FLAGS.result_dir,
        demo_data_label=FLAGS.demo_data_label,
        blackbox_optimization=FLAGS.blackbox_optimization_mode,
        fixed_workload=FLAGS.fixed_workload
    )

    assert FLAGS.generate_workload or FLAGS.restore_workload, "Either generate or import workload Flag must be True."
    train_queries, test_queries = None, None
    # Generate workload if needed
    if FLAGS.fixed_workload is True:
        workload = FixedIMDBWorkload()
        train_queries = workload.define_train_queries()
        if FLAGS.blackbox_optimization_mode is True:
            test_queries = workload.define_test_queries()
        else:
            # Pre-train from fixed human demonstration, test generalisation on other random queries
            # out of distribution.
            test_workload = IMDBSyntheticWorkload({ "num_selections": controller.schema.max_fields_per_index})
            test_queries = test_workload.define_test_queries(controller.queries_per_episode)
            print('Testing generalisation from fixed demos.')
        print("Got fixed {} train and {} test queries.".format(len(train_queries), len(test_queries)))

    else:
        # In black-box mode, do not generate, just import.
        if FLAGS.blackbox_optimization_mode is True:
            assert FLAGS.generate_workload is False, "Generate workload is True but must be False - " \
                                                     "Do not generate new training data in blackbox mode."
            assert FLAGS.restore_workload is True, "Restore workload is False but must be True -" \
                                                   "Do not generate new training data in blackbox mode."

        train_queries, test_queries = None, None
        # Generate workload if needed
        if FLAGS.generate_workload is True:
            print("Generating workload (blackbox mode = {})..".format(FLAGS.blackbox_optimization_mode))
            train_queries, test_queries = controller.generate_workload(export=True, label="imdb")
            print("Generated {} train and {} test queries.".format(len(train_queries), len(test_queries)))

        # Otherwise restore workload.
        if FLAGS.restore_workload is True:
            assert not train_queries
            assert not test_queries
            print("Restoring workload (blackbox mode = {})..".format(FLAGS.blackbox_optimization_mode))
            train_queries, test_queries = controller.restore_workload(train_label=FLAGS.train_label,
                                                                      test_label=FLAGS.test_label)
            print("Restored {} train and {} test queries.".format(len(train_queries), len(test_queries)))

    assert len(train_queries) > 0, "Need more than one train query but found 0."
    assert len(test_queries) > 0, "Need more than one train query but found 0."

    train_label = FLAGS.train_label
    test_label = FLAGS.test_label

    # Labels are concatenated like so:
    # train_label + workload mode + result
    # mysql_test_default_runtimes.csv

    # Call relevant evaluations on baselines.
    if FLAGS.evaluate_default:
        print("Evaluating default behaviour.")
        # Default does not need to create any indices, just reset.
        controller.reset_system()
        controller.evaluate(test_queries, label=test_label + "_default")

    # TODO evaluate all rules spearately.
    if FLAGS.evaluate_demo_rule:
        print("Evaluating demo rule.")
        demo_rule = MongoFullIndexing()
        actions = controller.act(test_queries=test_queries, demo_rule=demo_rule)
        index_performance = controller.evaluate(test_queries, label=test_label + "_demo_rule", actions=actions)
        controller.analyse_evaluation(index_performance, test_label + "_demo_rule")
        controller.reset_system()

    imported = False
    if FLAGS.evaluate_pretrain:
        print("Evaluating pretrained behaviour.")
        # Import a model, act on the model, evaluate, reset.
        assert not FLAGS.train_online_only
        controller.import_model(FLAGS.model_load_path)
        imported = True
        actions = controller.act(test_queries)
        index_performance = controller.evaluate(test_queries, label=test_label + "_pretrain", actions=actions)
        controller.analyse_evaluation(index_performance, test_label + "_pretrain")
        controller.reset_system()

    # Call training, evaluate after.
    if FLAGS.train_online_only:
        print("Evaluating online only (no pretraining) behaviour.")
        # No import, just train, act, evaluate, reset.
        # Train resets after every episode.
        controller.train(train_queries, label=train_label + "_online_only_")
        print("Finished training, beginning final evaluation.")
        controller.reset_system()
        if FLAGS.blackbox_optimization_mode is True:
            print("Evaluating best")
            best = controller.best_train_index_set
            actions = controller.act(test_queries, index_set=best)
        else:
            print("Evaluating learned model.")
            actions = controller.act(test_queries)

        index_performance = controller.evaluate(test_queries, label=test_label + "_online_only", actions=actions,
                                                test_similarity=True)
        controller.analyse_evaluation(index_performance, test_label + "_online_only")
        controller.reset_system()

    elif FLAGS.train_pretrain_online:
        print("Evaluating pretrain+online behaviour.")
        # Import if needed, train, act, evaluate, reset.
        if not imported:
            controller.import_model(FLAGS.model_load_path)
        controller.train(train_queries, label=train_label + "_pretrain_online_")
        controller.reset_system()
        print("Finished training, beginning final evaluation.")
        if FLAGS.blackbox_optimization_mode is True:
            print("Evaluating best")
            best = controller.best_train_index_set
            actions = controller.act(test_queries, index_set=best)
        else:
            print("Evaluating learned model.")
            actions = controller.act(test_queries)

        index_performance = controller.evaluate(test_queries, label=test_label + "_pretrain_online", actions=actions,
                                                test_similarity=True)
        controller.analyse_evaluation(index_performance, test_label + "_pretrain_online")
        controller.reset_system()


if __name__ == '__main__':
    main(sys.argv)
