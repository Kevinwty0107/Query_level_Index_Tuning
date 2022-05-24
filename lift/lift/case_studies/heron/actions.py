from subprocess import call
import argparse
# script to add items to a heron topology in python.
# works by running an external command.
def change_parallelism(cluster_role_env, name, component_pars, config_path=None):
    component_par_args = []
    for component_par in component_pars:
        component_par_args.append("--component-parallelism={}".format(component_par))
    if config_path is not None:
        component_par_args.append("--config-path={}".format(config_path))
    command = ["heron", "update", cluster_role_env, name]
    command.extend(component_par_args)
    return call(command)

def start_topology(cluster_role_env, path_to_jar, className, name, 
        load_dhalion=False, config_path='/local/scratch/be255/conf'):
    if load_dhalion:
        print('Loading Dhalion')
        command = ['heron', 'submit', '--config-property', 'heron.topology.healthmgr.mode=cluster', \
            '--config-property', 'heron.topology.metricscachemgr.mode=cluster', \
            '--config-path', config_path,
            cluster_role_env, path_to_jar, className, name]
    else:
        print('Not Loading Dhalion')
        command = ['heron', 'submit', cluster_role_env, path_to_jar, className,
                '--config-path', config_path,
                name]
    return call(command)

def main(argv):
    parser = argparse.ArgumentParser(description='Program to change the parallelism')
    parser.add_argument('cluster_role_env')
    parser.add_argument('topology')
    parser.add_argument('names_and_parallelisms')
    args = parser.parse_args()
    component_pars = args.names_and_parallelisms.split('/')
    change_parallelism(args.cluster_role_env, args.topology, component_pars)



# also would be nice to extend this to also be able
# to restart slow instances in a different container
# and also to change the operation of the hash function
# in order to reduce data skew
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
