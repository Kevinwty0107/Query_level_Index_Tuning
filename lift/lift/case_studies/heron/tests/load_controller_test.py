import sys
import argparse
import time
import json
import logging
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                          datefmt='%y-%m-%d %H:%M:%S')
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(sh)

from lift.case_studies.heron.load_controller import LoadController

def main(argv):
    parser = argparse.ArgumentParser('Test for Load Controller')
    parser.add_argument('experiment_config', help='example load_config')

    args = parser.parse_args()
    experiment_config = json.load(open(args.experiment_config, 'r'))
    load_controller = LoadController(experiment_config)
    # read what has been written to the queue
    load_controller.start()
    # add to the messages list
    messages = []
    while True:
        msg = load_controller.read()
        messages.append(msg)
        print(msg)
        if msg['word'] == -1:
            break
        time.sleep(0.01)
    # replay the message
    load_controller.reset()
    load_controller.start(replay=True)
    i = 0
    while True:
        msg = load_controller.read()
        print('Previous: {}'.format(messages[i]))
        print('Current: {}'.format(msg))
        if msg['word'] == -1:
            break
        i += 1
        time.sleep(0.05)
    print('Success!')

if __name__ == "__main__":
    main(sys.argv[1:])
