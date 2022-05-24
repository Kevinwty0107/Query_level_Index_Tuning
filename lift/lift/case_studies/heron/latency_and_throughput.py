import argparse 
import time
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from actions import change_parallelism
from heron_api import TrackerQueryGenerator, TrackerJSONParser
from metrics_collector import MetricsCollector
from scaler import ObjectiveScaler
from utils import gen_fixed_sum
"""

Simple program to collect latency and throughput measurements for 
a variety of different cluster configurations. Plots the results
on a scatter plot

"""
class ThroughputLatencyCollector:

    def __init__(self, filename, cluster, role, env, name, component_list,
                 wait_time = 90, collect_time=15, delay=60, 
                 config_path = None, write=False, verbose = False, sleep=60):
        """
            filename: e.g. throughput_and_latency.csv
            cluster_role_env: e.g. 'aurora/be255/devel'
            name: e.g. WordCountTopology
            component_list: e.g. ['sentence', 'split', 'count']
            wait_time (optional) : no of seconds to wait for the change
                to occur
            collect_time : no of minutes to collect stuff for.
        """
        self.file = open(filename, "a")
        # construct the string to write to the file
        if write:
            str_list = []
            for component in component_list:
                str_list.append(component)
            str_list.append('latency')
            str_list.append('throughput')
            csv = ','.join(str_list)
            csv += '\n'
            self.file.write(csv)
        # open the file
        self.cluster = cluster
        self.role = role
        self.env = env
        self.cluster_role_env = '{}/{}/{}'.format(cluster, role, env)
        self.name = name
        self.component_list = component_list
        self.wait_time = wait_time
        self.tqg = TrackerQueryGenerator()
        self.delay = delay
        self.collect_time = collect_time
        self.config_path = config_path
        self.verbose = verbose 
        self.sleep = sleep
        self.last_ack_count = 0
        self.max_retries = 5

    def _query(self):
        start_time = time.time()
        latency_url, latency_params = self.tqg.get_latency(self.cluster, 
            self.role, self.env, self.name, self.component_list[0],
            (int(start_time) - self.delay, int(start_time)))
        throughput_url, throughput_params = self.tqg.get_throughput(self.cluster,
            self.role, self.env, self.name, 
            self.component_list[len(self.component_list) - 1],
            (int(start_time) - self.delay, int(start_time)))
        latency_mc = MetricsCollector(latency_url, latency_params, 
            TrackerJSONParser.metric_parse, agg_fn=max, 
            result_fn=lambda x: x[0], verbose = self.verbose)
        throughput_mc = MetricsCollector(throughput_url, throughput_params,
            TrackerJSONParser.metric_parse, agg_fn=max,
            result_fn=lambda x: x[0], verbose = self.verbose, 
            metric='throughput_metric')
        latency = latency_mc.query()
        ack_count = throughput_mc.query()
        print(ack_count)
        if ack_count is not None:
            throughput = (ack_count[0] - self.last_ack_count) / self.delay
            self.last_ack_count = ack_count[0]
        else:
            throughput = None
        # append the results to the file
        return latency[0] if latency else None, throughput
                            
         
    
    def collect(self, par_dict):
        # collect the paralleism into a list
        component_par_list = []
        for component in self.component_list: 
            component_par_list.append('{}:{}'.format(component, 
                par_dict[component]))
        # change the parallelism
        failures = 0
        for i in range(self.max_retries):
            return_code = change_parallelism(self.cluster_role_env, self.name, 
                component_par_list, config_path = self.config_path)
            if return_code:
                print('Return Code {} is a failure -- trying again'.format(return_code))
                failures = failures + 1
            else:
                break
        if failures == self.max_retries - 1:
            return

        # wait for a certain amount of time
        time.sleep(self.wait_time)
        # collect throughput and latency for 
        # some time while ensuring to avoid
        # clock drift
        start_time = time.time()
        latencies = np.zeros(self.collect_time, dtype=np.float64)
        throughputs = np.zeros(self.collect_time, dtype=np.float64)
        counter = self.collect_time
        while counter > 0:
            latency, throughput = self._query()
            if not latency or not throughput:
                # sleep for some time and try again
                print('Collecting values failed -' +\
                        'sleeping for {} seconds'.format(self.sleep))
                time.sleep(self.sleep)
                continue
            latencies[-counter] = latency
            throughputs[-counter] = throughput
            
            # write the results to file
            str_list = []
            for component in self.component_list:
                str_list.append(str(par_dict[component]))
            str_list.append(str(latencies[-counter]))
            str_list.append(str(throughputs[-counter]))
            csv = ','.join(str_list)
            csv += '\n'
            self.file.write(csv)
            # sleep until we need to check again
            time.sleep(60.0 - ((time.time() - start_time) % 60))
            counter = counter - 1
        return latencies, throughputs
 
def stddev(x):
    return x.std() / np.sqrt(x.count())

def main():
    # create a latency and throughput class   
    parser = argparse.ArgumentParser(\
        description='Program to measure throughput and latency of heron')
    parser.add_argument('--dry-run', action = 'store_true')
    parser.add_argument('cluster_role_env')
    parser.add_argument('topology')
    parser.add_argument('--runs')
    parser.add_argument('--max-instances')
    parser.add_argument('--verbose', action= 'store_true')
    parser.add_argument('--alpha')
    parser.add_argument('--config-path')
    # plots an objective of the form (1 - x)L + xT where L is the scaled
    # and normalised latency and T is the scaled and normalised throughput
    parser.add_argument('--combined-objective', action = 'store_true')
    # just plot the scatter plot from the saved data
    parser.add_argument('--plot-scatter', action = 'store_true')
    # just plot the histogram from the saved data
    parser.add_argument('--plot-histogram', action = 'store_true')
    parser.add_argument('--combine-estimates', action = 'store_true')
    parser.add_argument('--write', action = 'store_true')
    args = parser.parse_args()
    if args.combined_objective and args.plot_scatter:
        raise RuntimeError('Current combination of arguments not supported')
    components = ['sentence', 'split', 'count']
    # parse the cluster_role_env string
    cluster_role_env = args.cluster_role_env
    cluster_role_env = cluster_role_env.split('/')
    if len(cluster_role_env) != 3:
        print('Must define a separate cluster, role and env.')
        return
    cluster = cluster_role_env[0]
    role = cluster_role_env[1]
    env = cluster_role_env[2]
    no_layers = len(components) 
    topology = args.topology

    print('Cluster/Role/Env: {}/{}/{}'.format(cluster, role, env))
    print('Topology: {}'.format(topology))
    
    if not args.dry_run:
        collect_time = 8
        collector = ThroughputLatencyCollector('through_latency.csv', 
                cluster, role, env, topology,
            components, collect_time = collect_time, 
            config_path=args.config_path, verbose = args.verbose,
            write = args.write)
        divide_by = np.sqrt(collect_time)
        #max_par = 2
    else:
        no_samples = 10000
        divide_by = np.sqrt(no_samples)
    # create the figure and axis
    #plt.ion()

    if args.runs:
        max_runs = int(args.runs)
    else:
        max_runs = 25

    if args.max_instances:
        max_instances = args.max_instances
    else:
        max_instances = 16

    if args.alpha:
        alpha = float(args.alpha)
    else:
        alpha = 0.5


    if args.combined_objective or args.plot_scatter or args.plot_histogram:
        # load through_latency
        df = pd.read_csv('through_latency.csv')
        # aggregate and compute the mean
        df = df.groupby(components, as_index=False).agg(
                {'latency':['mean', stddev, 'count'], 
                'throughput':['mean', stddev, 'count']})
        if args.combined_objective or args.plot_histogram:
            # compute the scalers 
            latency_scaler = ObjectiveScaler(
                    df['latency']['mean'],
                    df['latency']['stddev'], 
                    df['latency']['count'], args.combine_estimates, 
                    args.verbose)
            throughput_scaler = ObjectiveScaler(
                    df['throughput']['mean'],
                    df['throughput']['stddev'],
                    df['throughput']['count'], args.combine_estimates,
                    args.verbose)
            objectives = []

        # actually plot the latency and throughputs
        if args.plot_scatter:
            latency = df['latency']['mean']
            latency_std = df['latency']['stddev']
            throughput = df['throughput']['mean']
            throughput_std = df['throughput']['stddev']
            fig = plt.figure()
            plt.errorbar(latency, throughput, xerr=latency_std,
                yerr=throughput_std, fmt='+')
            plt.show()
            #fig.savefig('latency_and_throughput.pdf')
            return

        if args.plot_histogram:
            latencies = df['latency']['mean']
            throughputs = df['throughput']['mean']
            # scale the latencies and throughputs
            latencies = latencies.apply(latency_scaler.scale)
            throughputs = throughputs.apply(throughput_scaler.scale)
            objective = -alpha * latencies + (1 - alpha) * throughputs
            if args.verbose:
                print('Latencies: {}'.format(latencies))
                print('Throughputs: {}'.format(throughputs))
                print('Objective: {}'.format(objective))
                print(df.head)
            # plot the objective
            fig = plt.figure()
            plt.hist(objective)
            plt.xlabel('Objective')
            plt.ylabel('Frequency')
            # plot the latency
            fig1 = plt.figure()
            plt.hist(-latencies)
            plt.xlabel('Latency')
            plt.ylabel('Frequency')
            # plot the throughput
            fig2 = plt.figure()
            plt.hist(throughputs)
            plt.xlabel('Throughput')
            plt.ylabel('Frequency')
            plt.show()
            #fig.savefig('objective.pdf')
            return

    while max_runs > 0:
        # generate random parallelisms
        if not args.dry_run:
            # construct random numbers that sum to a number less than
            # or equal to max_instances
            random_pars = gen_fixed_sum(max_instances, 
                    len(components) + 1)[:len(components)]
            par_dict = dict()
            for i in range(len(components)):
                par_dict[components[i]] = random_pars[i]
            if args.verbose:
                print('Argument Parallelism: {}'.format(par_dict))
            latencies, throughputs = collector.collect(par_dict)
        else:
            latencies = np.random.rand(no_samples)
            throughputs = np.random.rand(no_samples)
        # apply some function to average these results
        latency = np.mean(latencies)
        throughput = np.mean(throughputs)
        if not args.combined_objective:
            latency_std = np.std(latencies, ddof=1) / divide_by
            throughput_std = np.std(throughputs, ddof=1) / divide_by
            plt.errorbar(latency, throughput, xerr=latency_std,
                yerr=throughput_std)
            plt.pause(0.000000001)
        else:
            # scale the throughput and latency appropriately
            latency = latency_scaler.scale(latency)
            throughput = throughput_scaler.scale(throughput)
            # compute the combined objective
            objective = alpha * latency + (1 - alpha) * throughput
            # plot it somehow (maybe as a histogram)
            objectives.append(objective)
            plt.hist(objectives)
        max_runs -= 1
    plt.pause(10000000)
    
if __name__ == "__main__":
    main()
