import sys
import argparse
import json
from lift.case_studies.heron import heron_load_generators

def main(argv):
    # lookup the load_generator
    parser = argparse.ArgumentParser('Program to test the load_generators')
    parser.add_argument('config')

    args = parser.parse_args()

    cfd = open(args.config, 'r')
    config = json.load(cfd)
    parallelism = {'sentence' : 3}
    load_generator = heron_load_generators[config['load_generator']](
            config['load_config'], parallelism)
    for load in load_generator.loads():
        print(load)
    print('DONE!')

if __name__ == "__main__":
    main(sys.argv[1:])

