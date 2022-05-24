import sys
import argparse
import time
import logging

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s',
                                          datefmt='%y-%m-%d %H:%M:%S')
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
logPath = '/home/be255/ProjectWork/PartIII/logs/'
fileName = 'heron_system_model_test.log'
sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.FileHandler('{0}/{1}'.format(logPath, fileName))
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)


from lift.case_studies.heron.heron_system_model import HeronSystemModel

def main(argv):
    parser = argparse.ArgumentParser('Test Application for the system model')
    parser.add_argument('cluster_role_env')
    parser.add_argument('topology') 
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--print-json', action='store_true')
    args = parser.parse_args()
    cluster_role_env = args.cluster_role_env
    cluster_role_env = cluster_role_env.split('/')
    if len(cluster_role_env) != 3:
        print('Must define a separate cluster, role and env.')
        return
    cluster = cluster_role_env[0]
    role = cluster_role_env[1]
    env = cluster_role_env[2]
    topology = args.topology
    # hard coded to keep it simple but this is bad
    components = ['sentence', 'split', 'count']
    parallelisms = {'sentence': 2, 'split': 2, 'count' : 2}
    log_path = '/home/be255/ProjectWork/PartIII/data/logs'
    # create the system model
    system_model = HeronSystemModel(cluster, role, env, topology,
            parallelisms, log_path=log_path, verbose=args.verbose,
            delay = 5, 
            print_json=args.print_json)
    # observe this bad boy
    result = system_model.observe_system()
    if result is None:
        raise RuntimeError('No Result')
    action = {'sentence' : 4, 'split' : 1, 'count' : 2} 
    system_model.act(action) 
    # observe the system again
    delay = 60
    print(
        'Successfully performed an action -- waiting {} seconds'.format(delay))
    time.sleep(delay)
    system_model.observe_system()
    action = {'count' : 2, 'sentence' : 2, 'split' : 2}
    system_model.act(action)


if __name__ == "__main__":
    main(sys.argv[1:])
