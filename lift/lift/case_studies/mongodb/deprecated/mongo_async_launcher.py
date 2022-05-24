import sys
import logging
import json
from lift.case_studies.mongodb.deprecated.mongo_async_controller import MongoAsyncController
import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string('host', 'hasu', 'MongoDB host')
gflags.DEFINE_string('agent', '', 'Agent configuration .json')
gflags.DEFINE_string('network', '', 'Network configuration json')
gflags.DEFINE_string('schema', '', 'Schema configuration json')
gflags.DEFINE_string('experiment_config', '', 'General experiment configuration json')
gflags.DEFINE_string('result_dir', '', 'Path to result directory, end with /')
gflags.DEFINE_integer('duration', 3600, 'Experiment duration in seconds')
gflags.DEFINE_string('executor', 'random', 'Executor to use')
gflags.DEFINE_string('model_store_path', '', 'Path to load model checkpoint')
gflags.DEFINE_string('model_load_path', '', 'Path to store model checkpoint')
gflags.DEFINE_string('data', '', 'Path to data in case of learning from demonstrations.')
gflags.DEFINE_string('serialization_path', '', 'Path to store serialization results')

gflags.DEFINE_boolean('load', False, 'Whether to load a trained model')
gflags.DEFINE_boolean('store', False, 'Whether to store the final model')
gflags.DEFINE_boolean('serialize', False, 'Whether to serialize ')


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
    with open(FLAGS.network, 'r') as fp:
        network_config = json.load(fp=fp)

    service = MongoAsyncController(
        agent_config=agent_config,
        network_config=network_config,
        experiment_config=experiment_config,
        schema_config=schema_config,
        host=FLAGS.host,
        result_dir=FLAGS.result_dir,
        duration=FLAGS.duration,
        executor=FLAGS.executor,
        model_store_path=FLAGS.model_store_path,
        model_load_path=FLAGS.model_load_path,
        data_path=FLAGS.data,
        store_model=FLAGS.store,
        load_model=FLAGS.load,
        serialize=FLAGS.serialize,
        serialization_path=FLAGS.serialization_path
    )
    service.start()


if __name__ == '__main__':
    main(sys.argv)