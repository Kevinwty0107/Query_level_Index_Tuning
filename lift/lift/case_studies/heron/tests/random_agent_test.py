import sys
import argparse
import json
import logging
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                          datefmt='%y-%m-%d %H:%M:%S')
logger = logging.getLogger('')
logger.setLevel(logging.FATAL)

sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(sh)



from lift.case_studies.heron.heron_online_controller import HeronSystemController

def main(argv):
    parser = argparse.ArgumentParser("Random Agent Test: runs"
            + "random agent online")
    parser.add_argument('experiment_config', 
        help='json file describing the experiment')
    
    args = parser.parse_args()
    fd = open(args.experiment_config, 'r')
    logger.debug(fd)
    experiment_config = json.load(fd)
    logger.debug(experiment_config)
    agent_config = json.load(open('conf/random_agent_config.json', 'r'))
    controller = HeronSystemController(agent_config, None, experiment_config)
    # start the controller
    controller.run()
            


if __name__ == "__main__":
    main(sys.argv[1:])
