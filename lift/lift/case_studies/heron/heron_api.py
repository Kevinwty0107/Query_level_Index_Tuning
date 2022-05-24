# This code is heavily inspired by Twitter's Heron
# UI tool and allows querying of several
# common metrics.
import time
import re
import sys
import argparse
import logging
import numpy as np

from lift.case_studies.heron.metrics_collector import MetricsCollector


# Constants representing some URLs
TOPOLOGIES_URL_FMT = "{}/topologies"
LOGICALPLAN_URL_FMT = "{}/logicalplan".format(TOPOLOGIES_URL_FMT)
PHYSICALPLAN_URL_FMT = "{}/physicalplan".format(TOPOLOGIES_URL_FMT)

METRICS_URL_FMT = "{}/metrics".format(TOPOLOGIES_URL_FMT)
METRICS_QUERY_URL_FMT = "{}/metricsquery".format(TOPOLOGIES_URL_FMT)
METRICS_TIMELINE_URL_FMT = "{}/metricstimeline".format(TOPOLOGIES_URL_FMT)

# Queries below from 
# https://github.com/twitter/heron/blob/master/heron/tools/common/src/python/access/heron_api.py

capacity = "DIVIDE(" \
           "  DEFAULT(0," \
           "    MULTIPLY(" \
           "      TS({0},*,__execute-count/default)," \
           "      TS({0},*,__execute-latency/default)" \
           "    )" \
           "  )," \
           "  60000000000" \
           ")"

failures = "DEFAULT(0," \
           "  DIVIDE(" \
           "    TS({0},*,__fail-count/default)," \
           "    SUM(" \
           "      DEFAULT(1, TS({0},*,__execute-count/default))," \
           "      DEFAULT(0, TS({0},*,__fail-count/default))" \
           "    )" \
           "  )" \
           ")"

cpu = "DEFAULT(0, TS({0},*,__jvm-process-cpu-load))"

memory = "DIVIDE(" \
         "  DEFAULT(0, TS({0},*,__jvm-memory-used-mb))," \
         "  DEFAULT(1, TS({0},*,__jvm-memory-mb-total))" \
         ")"

gc = "RATE(TS({0},*,__jvm-gc-collection-time-ms))"


backpressure = "DEFAULT(0, TS(__stmgr__,*," \
               "__time_spent_back_pressure_by_compid/{0}))"
# measured as the emit-count per second of the final bolt
#throughput = "DIVIDE(" \
#        "   SUM(DEFAULT(0, TS({0},*,__ack-count/default))),"\
#        "   SUBTRACT({2}, {1})"\
#        ")"
throughput = '__ack-count/default'
#throughput = "DEFAULT(0, TS({0},*,__ack-count/default))"
# measured by the spout as the complete latency (i.e. the
# time from the start to the tuple being acked)
latency_query = "DEFAULT(0, TS({0},*,__complete-latency/default))"
latency = '__complete-latency/default'
queries = dict(
        cpu=cpu,
        capacity=capacity,
        failures=failures,
        memory=memory,
        gc=gc,
        backpressure=backpressure,
        throughput=throughput,
        latency=latency_query,
        latency_metric = latency,
        throughput_metric = throughput
)

""" 
Generates query URLs + argument dictionaries given some standard arguments
for use with the heron tracker REST API. These can be 
passed to a MetricsCollector to query the metric.
"""
class TrackerQueryGenerator:
   
    def __init__(self, tracker_url = "http://kiku.cl.cam.ac.uk:8888"):
        self.logger = logging.getLogger(__name__)
        self.tracker_url = tracker_url

    def format_url(self, url):
        return url.format(self.tracker_url)
    
    def get_physical_plan(self, cluster, role, environ, topology):
        params = dict(cluster = cluster,
                role = role,
                environ = environ,
                topology = topology)
        url = self.format_url(PHYSICALPLAN_URL_FMT)
        return ([url], [params])
    
    def get_logical_plan(self, cluster, role, environ, topology):
        params = dict(cluster=cluster,
                environ=environ,
                topology=topology)
        url = self.format_url(LOGICALPLAN_URL_FMT)
        return ([url], [params])
    

    def get_metrics_query(self, cluster, role, environ, topology,
            time_range, query, do_query=True, component=None, interval=-1):
        if do_query:
            params = dict(cluster = cluster,
                environ = environ,
                role = role,
                topology = topology,
                starttime = time_range[0],
                endtime=time_range[1],
                query=query)
            url = self.format_url(METRICS_QUERY_URL_FMT)
        else:
            params = dict(cluster=cluster,
                    environ=environ,
                    role=role,
                    topology = topology,
                    starttime = time_range[0],
                    endtime = time_range[1],
                    component=component,
                    metricname=query)
            if interval != -1:
                params['interval'] = interval
            url = self.format_url(METRICS_URL_FMT)
        
        return ([url], [params])
    
    def get_metric(self, cluster, role, environ, topology,
            component, time_range, metric):
        query = queries.get(metric).format(component)
        return self.get_metrics_query(cluster, role, environ,
                topology, time_range, query)
    # Need a separate function to call when querying 
    # backpressure since that can only be queried on
    # a per instance basis
    def get_capacity(self, cluster, role, environ, topology,
            component, time_range):
        query = queries.get('capacity').format(component)
        return self.get_metrics_query(cluster, role, environ, 
                topology, time_range, query)
        
    def get_latency(self, cluster, role, environ, topology,
            component, time_range, interval):
        #query = queries.get('latency').format(component)
        query = queries.get('latency_metric')
        return self.get_metrics_query(cluster, role, environ, 
                topology, time_range, query, component=component, 
                do_query=False, interval=interval)   
    
    def get_throughput(self, cluster, role, environ, topology,
            component, time_range):
        #query = queries.get('throughput').format(component,
        #            time_range[0], time_range[1])
        query = queries.get('throughput_metric')
        return self.get_metrics_query(cluster, role, environ, 
                topology, time_range, query, do_query=False, 
                component=component)
 
    def get_failures(self, cluster, role, environ, topology,
            component, time_range):
        query = queries.get('failures').format(component)
        return self.get_metrics_query(cluster, role, environ, 
                topology, time_range, query)

    def get_cpu(self, cluster, role, environ, topology,
             component, time_range):
        query = queries.get('cpu').format(component)
        return self.get_metrics_query(cluster, role, environ, 
                topology, time_range, query)

    def get_memory(self, cluster, role, environ, topology,
             component, time_range):
        query = queries.get('memory').format(component)
        return self.get_metrics_query(cluster, role, environ, 
                topology, time_range, query)

    def get_gc(self, cluster, role, environ, topology,
             component, time_range):
        query = queries.get('gc').format(component)
        return self.get_metrics_query(cluster, role, environ, 
                topology, time_range, query)

    def get_backpressure(self, cluster, role, environ, topology, 
            component, time_range):
        # query the instances
        instances_url, instances_params =  \
            self.get_physical_plan(cluster, role, environ, topology)
        mc = MetricsCollector(instances_url, instances_params,
                TrackerJSONParser.instance_parse)
        query = mc.query()
        self.logger.info('Instances: {}'.format(query))
        if query:
            instances = query[0]
        else: 
            return None
        urls = []
        params = []

        for instance in instances:
            query = queries.get('backpressure').format(instance)
            url, param = self.get_metrics_query(cluster, role, environ, 
                    topology, time_range, query)
            the_component = instance.split('_')[2]
            if component == the_component:
                urls.append(url[0])
                params.append(param[0])
        return (urls, params)

    

"""
Contains several functions for parsing the results of 
typical JSON objects returned by queries from the 
TrackerQueryGenerator into items that can be used as 
part of the state of the Stream Processing system.
"""
class TrackerJSONParser:
    
    def __init__(self):
        self.throughput_ack_counts = []
    
    def reset(self):
        self.throughput_ack_counts = []

    @staticmethod
    def instance_parse(json_dict, agg_fn = lambda x: x):
        return json_dict['instances'].keys()

    @staticmethod
    def parallelism_parse(json_dict, agg_fn = lambda x: x):
        spouts = json_dict['spouts']
        bolts = json_dict['bolts']
        parallelism = dict()
        for spout, instances in spouts.items():
            parallelism[spout] = len(instances)
        for bolt, instances in bolts.items():
            parallelism[bolt] = len(instances)
        return parallelism

    @staticmethod
    def print_parse(json_dict, agg_fn = lambda x: x):
        print(json_dict)
    
    # agg_fn is a function to use to 
    # aggregate the results from different
    # parallel instances
    # name_to_index -- dictionary mapping from
    # 
    @staticmethod
    def metric_query_parse(json_dict, agg_fn = lambda x: x, 
            name_to_index=None, delay = 60):
        #starttime = int(json_dict['starttime'])
        #endtime = int(json_dict['endtime'])
        # compute the first time in the
        # dictionary
        #starttime = (starttime // delay + 1) * delay
        results = []
        for instance_dict in json_dict['timeline']:
            data_dict = instance_dict['data']
            if not data_dict:
                return None
            times = sorted([int(k) for k, _ in data_dict.items()], reverse=True)
            value = 0.0
            for time in times:
                if data_dict[str(time)] != 0.0:
                    value = data_dict[str(time)]
             
            results.append(value)
        return agg_fn(results)
        # TODO change so that the most recent one is 
        # used instead of the oldest one 
        #for time in range(starttime, endtime, delay):
            # query the point in time for all
            # returned instances
        #    time_list = []
        #    for instance_dict in json_dict['timeline']:
        #        data_dict = instance_dict['data']
        #        if str(time) not in data_dict:
        #            return None
                # get each time 
        #j        time_list.append(data_dict[str(time)])
        #    results.append(time_list)
        #ret = []
        #for time_list in results:
        #    # check whether time_list is empty
        #    if not time_list:
        #        return None
        #    ret.append(agg_fn(time_list))
        #return ret
    
        
    @staticmethod
    def metric_parse(json_dict, agg_fn = lambda x: x, metric = 'latency_metric'):
        metrics = json_dict['metrics']
        # get the particular metric that we are expecting
        the_metric = queries.get(metric)
        if the_metric not in metrics:
            return None
        results = [float(x) for x in metrics[the_metric].values()]
        # apply the aggregation function to these
        return [agg_fn(results)]
   
    @staticmethod
    def throughput_parse(json_dict, agg_fn = lambda x: x, 
            previous_counts = None):
        metrics = json_dict['metrics']
        the_metric = queries.get('throughput_metric')
        if the_metric not in metrics:
            return None
        interval = float(json_dict['interval'])
        measured_counts = [float(x) for x in metrics[the_metric].values()]
        results = [x / interval for x in measured_counts]
        return [agg_fn(results)]

    @staticmethod
    def backpressure_parse(json_dict, agg_fn = lambda x: x, delay=60):

        time_list = []
        timeline = json_dict['timeline']
        if not timeline:
            return None
        data_dict = timeline[0]['data']
        time = max(int(k) for k, _ in data_dict.items()) 
        time_list.append(data_dict[str(time)])
        return agg_fn(time_list)
    
    
    @staticmethod
    def aggregate_list_function(list_of_lists, the_fn = max):
        #Aggregates a list by applying a function across it.
        #    Really want a function that aggregates across instances.
        #    Thus will have to fix this before putting it into a state.
        if not list_of_lists or not isinstance(list_of_lists[0], list):
            return list_of_lists
        result = []
        for l in list_of_lists:
            result.append(the_fn(l))
        return result

    @staticmethod
    def logical_plan_parse(json_dict, agg_fn = lambda x: x):
        # get the number of instances
        no_nodes = len(json_dict['bolts']) + len(json_dict['spouts'])
        # adjacency matrix
        adj = np.full((no_nodes, no_nodes), False)
        # dictionary mapping from component names to adjacency matrix
        # indices
        name_to_index = dict()
        index = 0
        spouts = json_dict['spouts'].keys()
        bolts = json_dict['bolts'].keys()
        for component in json_dict['spouts'].keys():
            name_to_index[component] = index
            index += 1
        for component in json_dict['bolts'].keys():
            name_to_index[component] = index
            index += 1
        assert index == no_nodes
        # build adjacency matrix
        for component, connections in json_dict['bolts'].items():
            # get the inputs
            inputs = connections['inputs']
            # iterate through the inputs
            for info in inputs:
                name = info['component_name']
                adj[name_to_index[name], name_to_index[component]] = True
        return name_to_index, adj, list(spouts), list(bolts)


def main(argv):
    tqg = TrackerQueryGenerator()
    lag = 60
    parser = argparse.ArgumentParser(
            description='Query program for heron-tracker API')
    parser.add_argument('cluster_role_env')
    parser.add_argument('topology')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--print-json', action='store_true')
    args = parser.parse_args(argv)
    
    cluster_role_env = args.cluster_role_env.split('/')
    if len(cluster_role_env) != 3:
        print('Must define a cluster, role and environment')
        return

    cluster = cluster_role_env[0]
    role = cluster_role_env[1]
    env = cluster_role_env[2]
    topology = args.topology
    
    url, params = tqg.get_physical_plan(cluster, role, env, topology)
    mc_2 = MetricsCollector(url, params, TrackerJSONParser.parallelism_parse,
            agg_fn = lambda x: x, result_fn = lambda x: x[0], verbose=args.verbose,
            print_json=args.print_json)
    result = mc_2.query()
    
    print('Parallelism: {}'.format(result))

    url_4, params_4 = tqg.get_throughput(cluster, role, env, topology,
            'count', (int(time.time()) - lag, int(time.time())))
    mc_2 = MetricsCollector(url_4, params_4, TrackerJSONParser.print_parse,
            agg_fn = max, result_fn = lambda x: x[0], verbose = args.verbose,
            print_json = args.print_json)
    result = mc_2.query()
    print('Throughput: {}'.format(result))


    logical_plan_url, logical_plan_params = tqg.get_logical_plan(cluster,
            role, env, topology)
    logical_plan_mc = MetricsCollector(logical_plan_url, logical_plan_params,
            TrackerJSONParser.logical_plan_parse, agg_fn = lambda x: x, 
            result_fn = lambda x: x[0], verbose = args.verbose, 
            print_json = args.print_json)
    result = logical_plan_mc.query()
    print(result)
    name_to_index = {'word': 0, 'consumer': 1}

    url_and_params = tqg.get_backpressure(
            cluster, role, env, topology,
            (int(time.time()) - lag, int(time.time())), 
            name_to_index=name_to_index)
    if url_and_params:
        url, params = url_and_params
        mc = MetricsCollector(url, params, TrackerJSONParser.backpressure_parse, 
            result_fn = TrackerJSONParser.aggregate_list_function, 
            verbose = args.verbose, print_json = args.print_json)
        returned = mc.query()
        print('BackPressure: {}'.format(returned))
 

    url_2, params_2 = tqg.get_gc(cluster, role, env, topology,
            'word', (int(time.time()) - 3* lag, int(time.time())))
    
    mc_2 = MetricsCollector(url_2, params_2, TrackerJSONParser.metric_query_parse, 
            agg_fn = max, result_fn = lambda x: x[0], verbose=args.verbose,
            print_json = args.print_json)
    result = mc_2.query()
    print('GC: {}'.format(result))

    url_2, params_2 = tqg.get_latency(cluster, role, env, topology,
            'sentence', (int(time.time()) - lag, int(time.time())))
    mc_2 = MetricsCollector(url_2, params_2, TrackerJSONParser.metric_parse,
            agg_fn = max, result_fn = lambda x: x[0], 
            metric = 'latency_metric', verbose = args.verbose,
            print_json = args.print_json)
    result = mc_2.query()
    print('Latency: {}'.format(result))    

    url_4, params_4 = tqg.get_throughput(cluster, role, env, topology,
            'count', (int(time.time()) - lag, int(time.time())))
    mc_2 = MetricsCollector(url_4, params_4, TrackerJSONParser.throughput_parse,
            agg_fn = max, result_fn = lambda x: x[0], verbose = args.verbose,
            print_json = args.print_json)
    result = mc_2.query()
    print('Throughput: {}'.format(result))

if __name__=="__main__":
    main(sys.argv[1:])
