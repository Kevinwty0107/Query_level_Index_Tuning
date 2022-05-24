import logging
import time
import numpy as np
import json
from lift.model.state import State
from lift.model.action import Action
from lift.case_studies.mongodb.deprecated.execution import SystemModel
from lift.case_studies.heron.heron_api import \
        TrackerQueryGenerator, TrackerJSONParser
from lift.case_studies.heron.metrics_collector import MetricsCollector
from lift.case_studies.heron.actions import change_parallelism

class HeronSystemModel(SystemModel):

    def __init__(self, cluster, role, env, name, parallelisms, 
            acceptable_misses, failures = 4, delay = 10, wait_time=60, log_path = None,
            config_path = None, verbose = False, print_json=False, 
            max_instances = 15, reward_sleep_time = 15, max_over_instances = True):
        super(HeronSystemModel, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.print_json = print_json
        self.cluster = cluster
        self.role = role
        self.env = env
        self.name = name
        self.components = list(parallelisms.keys())
        self.tqg = TrackerQueryGenerator()
        self.failures = failures
        self.reward_sleep_time = reward_sleep_time
        self.max_over_instances = max_over_instances
        # records the constant state
        self.constant_state = dict()
        self.constant_state['max_instances'] = max_instances
        self.max_instances = max_instances
        # records the transient state
        self.observations = dict()
        self.constant_state['delay'] = delay
        self.constant_state['wait_time'] = wait_time
        self.config_path = config_path
        if log_path is not None:
            # open the file
            self.logger.info("Opening data logging file.")
            self.file = open(log_path, 'a')
            self.logger.info("Data logging file opened.")
        else:
            self.file = None
        self.metrics_collectors = dict()
        # query the logical plan
        self._init_logical_plan()
        #self.parallelisms = np.zeros((len(self.components),),dtype=np.int32)
        # also sets self.parallelisms
        self.original_parallelisms = self.get_parallelisms()
        self.logger.debug('Parallelisms: {}'.format(parallelisms))
        self.logger.debug('Name To Index: {}'.format(self.name_to_index))
        #for component, par in parallelisms.items():
        #    self.parallelisms[self.name_to_index[component]] = par
        #self.original_parallelisms = np.array(self.parallelisms, copy=True)
        # initialise all the metrics collectors
        agg_fn = max if self.max_over_instances else lambda x: x
        self.metrics_collectors['latency'] = MetricsCollector(None, None, 
            TrackerJSONParser.metric_parse, agg_fn=max, 
            result_fn=lambda x: x[0], verbose = self.verbose,
            print_json = self.print_json)
        self.metrics_collectors['ack_count'] = MetricsCollector(None, None,
            TrackerJSONParser.metric_parse, agg_fn=max,
            result_fn=lambda x: x[0], verbose = self.verbose, 
            metric='throughput_metric', print_json=self.print_json)
        # for several metrics with the same interface such as gc, cpu,
        # memory etc.
        self.metrics_collectors['metric'] = MetricsCollector(None, None,
            TrackerJSONParser.metric_query_parse, agg_fn=agg_fn,
            result_fn = lambda x: x[0], verbose = self.verbose,
            print_json = self.print_json, delay=self.constant_state['delay'])
        self.metrics_collectors['backpressure'] = MetricsCollector(None, None,
            TrackerJSONParser.backpressure_parse, 
            agg_fn=TrackerJSONParser.aggregate_list_function,
            result_fn = TrackerJSONParser.aggregate_list_function,
            verbose=self.verbose, print_json=self.print_json, 
            delay=self.constant_state['delay'])

        # metrics encompassed under the 'metric' banner
        self.common_metrics = ['cpu', 'gc', 'failures', 'memory', 'capacity']
        # names of everything to collect
        self.metric_names = ['cpu', 'gc', 'failures', 'memory', 
                'latency', 'ack_count', 'backpressure', 'capacity']
        # list of acceptable metrics misses 
        self.acceptable_misses = acceptable_misses

    def _cluster_role_env(self):
        return "{}/{}/{}".format(self.cluster, self.role, self.env)

    def _parallelism_to_list(self):
        ret = np.zeros((len(self.components),), dtype=np.int32)
        for component, index in self.name_to_index.items():
            ret[index] = self.parallelisms[component]
        return ret

    def _init_logical_plan(self):
        self.logger.info("Collecting logical plan...") 
        self.metrics_collectors['logical_plan'] = MetricsCollector(None,
                None, TrackerJSONParser.logical_plan_parse, agg_fn=lambda x: x,
                result_fn=lambda x: x[0], verbose = self.verbose)
        logical_plan_url, logical_plan_params = self.tqg.get_logical_plan(
                self.cluster, self.role, self.env, self.name)
        self.logger.debug('URL: {}'.format(logical_plan_url))
        self.logger.debug('Params:{}'.format(logical_plan_params))
        name_to_index, adj, spouts, bolts = \
                self.metrics_collectors['logical_plan'].query(logical_plan_url,
                    logical_plan_params)
        self.constant_state['name_to_index'] = name_to_index
        self.constant_state['adj'] = adj
        self.constant_state['spouts'] = spouts
        self.constant_state['bolts'] = bolts
        # just to save time
        self.name_to_index = self.constant_state['name_to_index']
        self.logger.info("...Logical plan collected")
    
    def is_bolt(self, component):
        return component in self.constant_state['bolts']
    
    def is_spout(self, component):
        return component in self.constant_state['spouts']

    def num_spouts(self):
        return len(self.constant_state['spouts'])

    def num_bolts(self):
        return len(self.constant_state['bolts'])
    
    def component_to_index(self, component):
        return self.name_to_index[component]
    
    def get_current_parallelisms(self):
        return self.parallelisms
    
    def get_parallelisms(self):
        url, params = self.tqg.get_physical_plan(self.cluster, self.role,
                self.env, self.name)
        mc = MetricsCollector(url, params, TrackerJSONParser.parallelism_parse,
                agg_fn = lambda x: x, result_fn = lambda x: x[0], 
                verbose = self.verbose)
        self.parallelisms = mc.query()
        return self.parallelisms 

    def get_constant_state(self):
        return self.constant_state

    def get_last_observation(self):
        return self.observations

    def observe_system(self, batch_size=0, to_collect=None):
        # get the urls/params for all the different metrics
        self.logger.info("Starting to observe the system")
        start_time = time.time()
        urls_and_params = dict()
        self.logger.info("Starting to generate URLs")
        # TODO make this more general to deal w/ multiple spouts
        urls_and_params['latency'] = self.tqg.get_latency(self.cluster, 
            self.role, self.env, self.name, self.constant_state['spouts'][0],
            (int(start_time) - self.constant_state['delay'], int(start_time)),
            self.constant_state['wait_time'])
        urls_and_params['ack_count'] = self.tqg.get_throughput(
            self.cluster, self.role, self.env, self.name, 
            'count', (int(start_time) - self.constant_state['delay'], 
                int(start_time)))
        backpressure_url_and_params = dict()
        for component in self.components:
            backpressure_url_and_params[component] = self.tqg.get_backpressure(
                    self.cluster, self.role, self.env, self.name, component, 
                    (int(start_time) - self.constant_state['delay'], 
                        int(start_time)))
        urls_and_params['backpressure'] = backpressure_url_and_params
        # Build the dictionaries for the other metrics
        for metric in self.common_metrics:
            # iterate over all the components in the list
            metric_url_and_params = dict()
            for component in self.components:
                metric_url_and_params[component] = self.tqg.get_metric(
                        self.cluster, self.role, self.env, self.name,
                        component, (int(start_time) - 
                            self.constant_state['delay'],
                            int(start_time)), metric)
            urls_and_params[metric] = metric_url_and_params
        self.logger.info("URLs generated, collecting metrics")
        # build observed metrics
        self.observations = dict()
        self.observations['metrics'] = dict()
        for metric_name in self.metric_names:
            if to_collect is not None and metric_name not in to_collect:
                continue
            if metric_name in self.common_metrics or metric_name == 'capacity':
                mc = self.metrics_collectors['metric']
            else:
                mc = self.metrics_collectors[metric_name]
            # check whether the metric name is in the list of generic
            # metrics or not
            self.logger.info("Collecting {} metric".format(metric_name))
            is_common_metric = \
                metric_name in self.common_metrics or \
                metric_name == 'backpressure'
            if is_common_metric:
                # TODO would need to change this to be a list
                # then can add lists to it 
                if self.max_over_instances:
                    results = np.zeros((len(self.components),))
                else:
                    results = [None] * len(self.components)
                metric_urls_and_params = urls_and_params[metric_name]
                for component in self.components:
                    urls, params = metric_urls_and_params[component]
                    self.logger.debug('URL:{}'.format(urls))
                    self.logger.debug('Params: {}'.format(params))
                    result = None
                    it = 0
                    while not result and it < self.failures and \
                            component not in self.acceptable_misses[metric_name]:
                        result = mc.query(urls, params)
                        it = it + 1
                    if component in self.acceptable_misses[metric_name]:
                        # need to fabricate some results
                        result = [0.0] * self.parallelisms[component]
                    if result:
                        # TODO To collect for all instances need to change this to
                        # accept list
                        if self.max_over_instances:
                            results[self.name_to_index[component]] = result[0]
                        else:
                            results[self.name_to_index[component]] = result
                    else:
                        return None
                # would now be a list of lists. TODO
                if self.max_over_instances:
                    self.observations['metrics'][metric_name] = results.tolist()
                else:
                    self.observations['metrics'][metric_name] = results
            else:
                urls, params = urls_and_params[metric_name]
                self.logger.debug('URL:{}'.format(urls))
                self.logger.debug('Params:{}'.format(params))
                result = None
                it = 0
                while not result and it < self.failures:
                    result = mc.query(urls, params)
                if result:
                    # TODO would be changed to be a list
                    
                    if np.isnan(result[0]):
                        return None
                    self.observations['metrics'][metric_name] = \
                        result[0]
                else:
                    return None
            self.logger.info(
                    "{} metric collected successfully".format(metric_name))
        self.observations['par'] = self._parallelism_to_list().tolist()
        self.observations['name_to_index'] = self.name_to_index
        self.logger.debug('System state: {}'.format(self.observations))
        # check whether to serialise to a file
        if self.file is not None:
            json.dump(self.observations, self.file)
        self.logger.info("Finished system observations")
        return State(self.observations)
    
    
    def act(self, action_obj, set_parallelism=True):
        action = action_obj.get_value()
        for spout in self.constant_state['spouts']:
            if action[spout] == -1:
                action[spout] = self.parallelisms[spout]
        for bolt in self.constant_state['bolts']:
            if action[bolt] <= 0:
                action[bolt] = 1
        self.logger.info(
                "Starting to perform system action, changing parallelism to" +
                "{} from {}".format(action, self.parallelisms))
        # This is a system action. This means that the 
        # save the state
        if self.is_noop(action):
            self.logger.info("Action was noop, returning")
            return
        # check whether we need to modify the action
        if sum(action.values()) > self.max_instances:
            # still a valid action ==> we can update the spout
            diff = sum(action.values()) - self.max_instances
            loss = diff // len(self.constant_state['bolts'])
            mod_loss = diff % len(self.constant_state['bolts'])
            for bolt in self.constant_state['bolts']:
                if mod_loss == 0:
                    action[bolt] = self.parallelisms[bolt] - loss 
                else:
                    action[bolt] = self.parallelisms[bolt] - loss - 1
                mod_loss -= 1
            # leave spouts untouched
        self.logger.info("After modification action is {}".format(action))
        # perform the action
        component_pars = []
        for component, parallelism in action.items():
            if parallelism < 0:
                parallelism = self.parallelisms[component]
            component_pars.append("{}:{}".format(component, parallelism))
        total_failures = 0
        for i in range(self.failures):
            return_code = change_parallelism(self._cluster_role_env(), 
                    self.name, component_pars, config_path = self.config_path)
            # did we fail
            if not return_code:
                break
            self.logger.info("Failed to change the parallelism,trying again")
            total_failures += 1
        
        if total_failures == self.failures:
            raise RuntimeError("Unable to perform the action requested")
        if set_parallelism:
            self.parallelisms = action

        self.logger.info("Parallelism successfully changed")

    def system_status(self):
        return self.observations

    def is_noop(self, action):
        if self.parallelisms[self.constant_state['spouts'][0]] != \
                action[self.constant_state['spouts'][0]]:
            return False
        # is it the same as the one we already have
        if self.parallelisms == action:
            return True
        # is the total larger than the max no_instances
        if sum(action.values()) > self.max_instances and \
                action[self.constant_state['spouts'][0]] == \
                self.parallelisms[self.constant_state['spouts'][0]]:
            return True
        # check whether the action only contains spouts and this is the 
        # same
        all_spouts = True
        for bolt in self.constant_state['bolts']:
            if bolt in action:
                all_spouts = False
        if all_spouts:

            for spout in self.constant_state['spouts']:
                if spout not in action:
                    return True
            for spout in self.constant_state['spouts']:
                if self.parallelisms[spout] != action[spout]:
                    return False
            return True

        # check whether updating the parallelisms for the spout would cause
        # us to go above the max no. of instances
        
        new_action = dict()
        for bolt in self.constant_state['bolts']:
            new_action[bolt] = self.parallelisms[bolt]
        for spout in self.constant_state['spouts']:
            new_action[spout] = action[spout]
        if sum(new_action.values()) > self.max_instances:
            return True
        return False

    def reset(self):
        # reset to the original parallelism
        self.logger.info("Resetting the environment")
        self.act(Action(self.original_parallelisms))
