import sys
import argparse
import json
import logging
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                          datefmt='%y-%m-%d %H:%M:%S')
logger = logging.getLogger('')
logger.setLevel(logging.INFO)

sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

logger.addHandler(sh)


from lift.case_studies.heron.heron_online_controller import HeronSystemController


def main(argv):
    parser = argparse.ArgumentParser("Heron Online Launcher: runs"
            " an agent online")
    parser.add_argument('experiment_config', help='json file describing'
            ' the experiment')
    parser.add_argument('agent_config', help='json file describing the agent')
    parser.add_argument('--load-path', help='directory to load the model from')
    parser.add_argument('--store-path', help='directory to save the model to')
    parser.add_argument('--test-only', action='store_true',
            help='Do not train the model -- only test')
    parser.add_argument('--network-config', help='json file containing the '
            'network configuration.')
    args = parser.parse_args()
    logger.debug('Parsing args:')

    if args.load_path:
        load_model = True
    else:
        load_model = False
    if args.store_path:
        store_model = True
    else:
        store_model = False
    efd = open(args.experiment_config, 'r')
    experiment_config = json.load(efd)
    if args.test_only:
        experiment_config['pretrain_serialise'] = False
    afd = open(args.agent_config, 'r')
    logger.debug(experiment_config)
    agent_config = json.load(afd)
    logger.debug(agent_config)
    if args.network_config:
        nfd = open(args.network_config, 'r')
        network_config = json.load(nfd)
    else:
        network_config = None

    logger.info('Creating heron controller:')
    controller = HeronSystemController(agent_config, network_config,
                                       experiment_config,
                                       load_model = load_model, store_model = store_model,
                                       model_store_path=args.store_path, model_load_path = args.load_path)
    if args.test_only:
        print('Only Testing')
    # start the controller
    logger.info('Starting heron controller:')
    controller.run(test_only=args.test_only)
            


if __name__ == "__main__":
    main(sys.argv[1:])
