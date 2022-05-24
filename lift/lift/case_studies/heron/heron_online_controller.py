import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorforce.agents import Agent

from lift.case_studies.heron import heron_model_generators, heron_schemas, \
    heron_reward_generators, heron_load_generators
from lift.case_studies.heron.heron_data_loader import PretrainDataSource
from lift.model.state import State
from lift.model.action import Action
from lift.case_studies.heron.heron_system_model import HeronSystemModel
from lift.case_studies.heron.scaler import ObjectiveScaler
from lift.case_studies.heron.actions import start_topology
from lift.controller.system_controller import SystemController
from lift.case_studies.heron.rule_based_agent import RuleBasedAgent
from lift.case_studies.heron.constant_agent import ConstantAgent
from lift.case_studies.heron.cheating_agent import CheatingAgent
from lift.case_studies.heron.reward_plotter import RewardPlotter


class HeronSystemController(SystemController):

    def __init__(
            self,
            agent_config,
            network_config,
            experiment_config,
            schema_config=None,
            result_dir='',
            model_store_path='',
            model_load_path='',
            data_path='',
            store_model=False,
            load_model=False
    ):
        super(HeronSystemController, self).__init__(agent_config,
                                                    network_config, experiment_config, schema_config,
                                                    result_dir, model_store_path, model_load_path,
                                                    data_path, store_model, load_model)
        self.cluster = experiment_config['cluster']
        self.role = experiment_config['role']
        self.env = experiment_config['env']
        self.cluster_role_env = '{}/{}/{}'.format(self.cluster, self.role,
                                                  self.env)
        self.topology = experiment_config['topology']
        self.parallelism = experiment_config['parallelism']
        self.components = list(self.parallelism.keys())
        self.delay = experiment_config['delay']
        self.wait_time = experiment_config['wait_time']
        self.fail_wait_time = experiment_config['fail_wait_time']
        self.failures = experiment_config['failures']
        self.load_dhalion = experiment_config['load_dhalion']
        # metrics used by the agent for state
        self.state_metrics = experiment_config['state_metrics']
        self.reward_metrics = experiment_config['reward_metrics']
        self.episodes = experiment_config['episodes']
        self.max_decrease = experiment_config['max_decrease']
        # create the system model
        self.path_to_jar = experiment_config['path_to_jar']
        self.class_name = experiment_config['class_name']
        self.save_dir = experiment_config['save_dir']
        self.full_state = experiment_config['converter'] == 'full_state'
        # TODO Implement the independent treatment of stages TODO.
        self.stages_independent = experiment_config['stages_independent']
        self.pretrain_serialise = experiment_config['pretrain_serialise']
        self.plot_rewards = experiment_config['plot_rewards']
        self.log_rewards = experiment_config['log_rewards']
        self.config_path = experiment_config['config_path']
        if self.plot_rewards:
            self.reward_plotter = RewardPlotter(
                experiment_config['reward_plot_file'])
        if self.pretrain_serialise:
            self.demo_dir = experiment_config['demo_dir']
            self.demo_file = experiment_config['pretrain_file']
        if self.log_rewards:
            self.results_dir = experiment_config['results_dir']
            self.result_file = experiment_config['results_file']
        # create the schema
        self.schema = heron_schemas[experiment_config['schemas']](
            experiment_config)
        self.logger.debug(self.schema)
        # create the latency scaler and throughput scaler
        # load the throughput and latency measurements from
        # a previous source. Initially this is a CSV, but
        # TODO is make this (state, action, reward) logs

        # create the network if there is no network config
        self.states_spec = self.schema.get_states_spec()
        self.action_spec = self.schema.get_actions_spec()
        if network_config:
            network_spec = network_config
        else:
            layer_size = experiment_config['layer_size']
            network_spec = [
                dict(type='flatten'),
                dict(type='dense', size=layer_size),
                dict(type='dense', size=layer_size)
            ]
        # Create the agent
        tf.reset_default_graph()
        if agent_config['type'] == 'random_agent':
            self.agent = Agent.from_spec(
                spec=agent_config,
                kwargs=dict(
                    states=self.states_spec,
                    actions=self.action_spec
                )
            )
        elif agent_config['type'] == 'constant':
            self.agent = ConstantAgent(agent_config)
        elif agent_config['type'] == 'cheating':
            self.agent = CheatingAgent(agent_config)
            self.load_demo_data = False
        elif agent_config['type'] == 'rules':
            self.agent = RuleBasedAgent(agent_config)
            self.load_demo_data = False
        else:
            self.agent = Agent.from_spec(
                spec=agent_config,
                kwargs=dict(
                    states=self.states_spec,
                    actions=self.action_spec,
                    network=network_spec
                )
            )
            # Use demo data when running online.
            self.load_demo_data = experiment_config.get('use_demo_data', True)

        if load_model:
            self.logger.info("Loading model from load path {}".format(model_load_path))
            self.agent.restore_model(model_load_path)
        self.store_model = store_model
        self.model_store_path = model_store_path
        tf.get_default_graph().finalize()
        # create the load generation
        # self.load_controller = LoadController(experiment_config)
        parallelism = experiment_config['parallelism']
        load_config = experiment_config['load_config']
        self.load_controller = heron_load_generators[experiment_config['load_generator']](
            load_config,
            parallelism
        )
        self.waiting_on_system_time = 0
        self.training_agent_time = 0
        self.total_time = 0
        # start the topology with the relevant configuration

    def _init_model_generator(self):
        df = pd.read_csv(self.experiment_config['reward_csv'])
        df['latency_sqrt'] = df['latency'].apply(np.sqrt)
        self.logger.info(df['latency_sqrt'])
        df = df.groupby(self.components, as_index=False).agg(
            {'latency_sqrt': ['mean', HeronSystemController._stddev, 'count'],
             'throughput': ['mean', HeronSystemController._stddev, 'count']})
        self.logger.debug(df.head())
        latency_scaler = ObjectiveScaler(
            df['latency_sqrt']['mean'],
            df['latency_sqrt']['_stddev'],
            df['latency_sqrt']['count']
        )
        throughput_scaler = ObjectiveScaler(
            df['throughput']['mean'],
            df['throughput']['_stddev'],
            df['throughput']['count']
        )
        # create the reward generator
        reward_generator = heron_reward_generators[ \
            self.experiment_config['reward_generator']](
            self.experiment_config['reward_generator_args'])
        # create the converter
        constant_state = self.system_model.get_constant_state()
        self.model_generator = heron_model_generators[ \
            self.experiment_config['converter']](
            constant_state, latency_scaler,
            throughput_scaler, reward_generator,
            self.experiment_config)

        # To preload demo data
        if self.load_demo_data:
            self.pretrain_loader = PretrainDataSource(self.experiment_config)
            self.trace_file = os.path.join(self.experiment_config['demo_dir'],
                                           self.experiment_config['demo_file'])

    @staticmethod
    def _stddev(x):
        return x.std() / np.sqrt(x.count())

    @staticmethod
    def extract_component(system_state, index, spout):
        state = dict()
        metrics_dict = system_state.as_dict()['metrics']
        state['metrics'] = dict()
        spout_index = system_state.as_dict()['name_to_index'][spout]
        for k, v in metrics_dict.items():
            if k == 'failures':
                state['metrics'][k] = v[spout_index]
            state['metrics'][k] = v[index]
        state['par'] = system_state.as_dict()['par'][index]
        state['spout_par'] = system_state.as_dict()['par'][spout_index]
        return State(state)

    def close(self):
        if self.pretrain_serialise:
            self.pretrain_file.close()

    def train(self, replay=False):
        # Load previous
        if self.load_demo_data:
            batch = self.pretrain_loader.load_data(self.trace_file, concat=False)
            self.agent.import_demonstrations(demonstrations=batch)

        # reset the system model
        throughputs = np.zeros((self.episodes,))
        latencies = np.zeros((self.episodes,))
        mean_rewards = np.zeros((self.episodes,))
        step = 0
        previous_system_reward = None
        for i in range(self.episodes):
            prev_ack_count = 0
            terminal = False
            time_start = time.perf_counter()
            self.system_model.reset()
            self.waiting_on_system_time += time.perf_counter() - time_start
            # do_replay = replay or i != 0
            rewards = []
            load_generator = self.load_controller.loads()
            total_latency = 0
            total_throughput = 0
            while not terminal:
                # query the state
                time_start = time.perf_counter()
                for j in range(self.failures):
                    system_state = self.system_model.observe_system(
                        to_collect=self.state_metrics)
                    if system_state:
                        break
                    time.sleep(self.fail_wait_time)
                if self.load_dhalion:
                    for j in range(self.failures):
                        self.parallelism = self.system_model.get_parallelisms()
                        if self.parallelism:
                            break

                if system_state == None:
                    raise RuntimeError('Cannot get system state, returning early')
                if self.load_dhalion and self.parallelism == None:
                    raise RuntimeError('Cannot fetch parallelism, returning early')
                self.waiting_on_system_time += time.perf_counter() - time_start
                # TODO if stages_independent is true, then extract the state
                # for a given component and ask the agent for an action. 
                # make sure independent = True so that the action need not be
                # followed by a reward. 
                if not self.stages_independent and not self.load_dhalion:
                    state = self.model_generator.system_to_agent_state(
                        system_state)
                    self.logger.info('Agent State: {}'.format(state.as_dict()))
                    agent_action = self.agent.act(states=state.as_dict())
                    self.logger.info('Agent Action: {}'.format(agent_action))
                    parallelism = self.system_model.get_current_parallelisms()
                    agent_action = self.model_generator.agent_to_system_action(
                        agent_action, parallelism)
                elif not self.load_dhalion:
                    # iterate over the bolts
                    it = 0
                    agent_actions = dict()
                    parallelism = self.system_model.get_current_parallelisms()
                    # get the spout
                    the_spout = \
                        self.experiment_config['load_config']['component']
                    for component in self.components:
                        if self.system_model.is_spout(component):
                            continue
                        index = self.system_model.component_to_index(component)
                        component_state = \
                            self.extract_component(system_state, index,
                                                   the_spout)
                        if self.full_state:
                            agent_state = \
                                self.model_generator.system_to_agent_state(
                                    component_state, add_to_scaler=True)
                        else:
                            agent_state = \
                                self.model_generator.system_to_agent_state(
                                    component_state)
                        independent = it != (self.system_model.num_bolts() - 1)
                        time_start = time.perf_counter()
                        agent_action = \
                            self.agent.act(states=agent_state.as_dict(),
                                           independent=independent)
                        self.training_agent_time += time.perf_counter() - \
                                                    time_start
                        agent_action['par'] += self.max_decrease
                        self.logger.info(
                            'Agent action for component {} was {}'.format(
                                component, agent_action)
                        )
                        # convert the agent action into a system action
                        agent_actions[component] = agent_action['par'] + \
                                                   parallelism[component]
                        it = it + 1
                # get the load_process action
                # TODO REMOVE ASSUMPTION THERE IS ONLY ONE SPOUT
                the_spout = self.experiment_config['load_config']['component']
                load_action = next(load_generator, {the_spout: -1})
                terminal = (list(load_action.values()) == [-1])
                # TODO merge the actions from the various stages of asking the
                # agent into a single system action.
                # merge these two dictionaries since keys are disjoint
                if not self.stages_independent and not self.load_dhalion:
                    action = {**(agent_action.get_value()), **load_action}
                elif not self.load_dhalion:
                    action = {**agent_actions, **load_action}
                elif self.load_dhalion:
                    action = load_action
                # perform this action
                time_start = time.perf_counter()
                self.system_model.act(Action(action),
                                      set_parallelism=(not self.load_dhalion))
                # wait a bit to ensure that the metrics are available
                time.sleep(self.wait_time)
                # TODO -- perhaps give the option to encapsulate it in 
                # a reward object
                # query the reward -- this will be a state object (woops)
                for k in range(self.failures):
                    system_reward = self.system_model.observe_system(
                        to_collect=self.reward_metrics)
                    if system_reward and system_reward.as_dict() and not \
                            np.isnan(system_reward.as_dict() \
                                             ['metrics']['latency']):
                        break
                    # sleep briefly
                    time.sleep(self.fail_wait_time)

                if not system_reward or not system_reward.as_dict() or \
                        np.isnan(system_reward.as_dict() \
                                         ['metrics']['latency']):
                    # simply return the previous reward
                    system_reward = previous_system_reward
                previous_system_reward = system_reward
                if not system_reward:
                    raise RuntimeError('Cannot get reward. Terminating.')
                self.waiting_on_system_time += time.perf_counter() - time_start
                reward = self.model_generator.system_to_agent_reward(
                    system_reward)
                # it a STATE -- this is weird but whatever TODO 
                system_reward = system_reward.states_dict()
                rewards.append(reward.get_value())
                self.logger.info('Reward: {}'.format(reward.get_value()))
                if self.plot_rewards:
                    self.reward_plotter.plot(step, reward.get_value())

                # call observe with appropriate state_value of terminal
                self.logger.info(terminal)
                time_start = time.perf_counter()
                self.agent.observe(terminal=terminal, reward=reward.get_value())
                self.training_agent_time += time.perf_counter() - time_start
                if self.pretrain_serialise:
                    # track the current parallelism
                    action = {**self.parallelism, **action}
                    self.pretrain_file.write(
                        '{}%{}%{}%{}\n'.format(system_state.as_dict(),
                                               action, system_reward, terminal))
                if self.log_rewards:
                    action = {**self.parallelism, **load_action}
                    self.logger.info("Writing reward to file..")
                    self.rewards_file.write('{}\n'.format(reward.get_value()))
                    self.rewards_file.flush()
                if self.full_state:
                    self.model_generator.serialise_scalers()
                # log latency and throughput
                total_latency += system_reward['metrics']['latency']
                total_throughput += \
                    (system_reward['metrics']['ack_count'] - \
                     prev_ack_count) / self.delay
                prev_ack_count = system_reward['metrics']['ack_count']
                step = step + 1
                # TODO log state etc. using metrics helper
                # TODO plot graph of the rewards over time 
            # print out a summary
            self.logger.info('----------------EPISODE-END-------------')
            self.logger.info('Total Latency: {}'.format(total_latency))
            self.logger.info(
                'Total Throughput: {}'.format(total_throughput))
            self.logger.info('Waiting on System Time: {}'.format(
                self.waiting_on_system_time)
            )
            self.logger.info('Training Agent Time: {}'.format(
                self.training_agent_time)
            )
            total_time_end = time.perf_counter()
            self.logger.info('Total Time (so far): {}'.format(
                total_time_end - self.start_run))
            # visualise reward
            rewards = np.asarray(rewards)
            mean_reward = np.mean(rewards)
            self.logger.info('Average Episode Reward: {}'.format(mean_reward))
            if self.store_model:
                # Don't save all steps.
                self.agent.save_model(self.model_store_path + '/training_model.ckpt', False)
            throughputs[i] = total_throughput
            latencies[i] = total_latency
            mean_rewards[i] = mean_reward

        # save the latencies and throughputs
        np.save(os.path.join(self.save_dir, 'throughputs.npy'), throughputs)
        np.save(os.path.join(self.save_dir, 'latencies.npy'), latencies)

    def test(self):
        terminal = False
        self.system_model.reset()
        rewards = []
        total_latency = 0
        total_throughput = 0
        prev_ack_count = 0
        load_generator = self.load_controller.loads()
        previous_system_reward = None
        while not terminal:
            for i in range(self.failures):
                system_state = self.system_model.observe_system(
                    to_collect=self.state_metrics)
                if system_state:
                    break
            if system_state == None:
                raise RuntimeError('Cannot get system state, returning early')
            if self.load_dhalion:
                for i in range(self.failures):
                    self.parallelism = self.system_model.get_parallelisms()
                    if self.parallelism:
                        break
                if not self.parallelism:
                    raise RuntimeError('Cannot fetch parallelism, returning')

            if not self.stages_independent and not self.load_dhalion:
                state = self.model_generator.system_to_agent_state(
                    system_state)
                agent_action = self.agent.act(states=state.as_dict(),
                                              independent=True, deterministic=True)
                parallelism = self.system_model.get_current_parallelisms()
                agent_action = self.model_generator.agent_to_system_action(
                    agent_action, parallelism)
            elif not self.load_dhalion:
                # iterate over the bolts
                the_spout = self.experiment_config['load_config']['component']
                agent_actions = dict()
                parallelism = self.system_model.get_current_parallelisms()
                for component in self.components:
                    if self.system_model.is_spout(component):
                        continue
                    index = self.system_model.component_to_index(component)
                    component_state = \
                        self.extract_component(system_state, index,
                                               the_spout)
                    agent_state = \
                        self.model_generator.system_to_agent_state(
                            component_state)
                    agent_action = \
                        self.agent.act(states=agent_state.as_dict(),
                                       independent=True, deterministic=True)
                    agent_action['par'] += self.max_decrease
                    # convert the agent action into a system action 
                    agent_actions[component] = agent_action['par'] + \
                                               parallelism[component]

            # get the load_process action
            the_spout = self.experiment_config['load_config']['component']
            load_action = next(load_generator, {the_spout: -1})
            terminal = (list(load_action.values()) == [-1])
            # merge these two dictionaries since keys are disjoint
            if not self.load_dhalion and not self.stages_independent:
                action = {**(agent_action.get_value()), **load_action}
            elif not self.load_dhalion:
                action = {**agent_actions, **load_action}
            else:
                action = load_action
            # perform this action
            self.system_model.act(Action(action))
            # wait a bit to ensure that the metrics are available
            time.sleep(self.wait_time)
            # query the reward
            for k in range(self.failures):
                system_reward = self.system_model.observe_system(
                    to_collect=self.reward_metrics)
                if system_reward and system_reward.as_dict() and not \
                        np.isnan(system_reward.as_dict() \
                                         ['metrics']['latency']):
                    break
                    # sleep briefly
                    time.sleep(self.fail_wait_time)
            if not system_reward or not system_reward.as_dict() or \
                    np.isnan(system_reward.as_dict() \
                                     ['metrics']['latency']):
                # simply return the previous reward

                system_reward = previous_system_reward
            previous_system_reward = system_reward
            if not system_reward:
                raise RuntimeError('Cannot get reward. Terminating.')

            reward = self.model_generator.system_to_agent_reward(
                system_reward)
            self.logger.info('Reward: {}'.format(reward.get_value()))
            rewards.append(reward.get_value())

            if self.pretrain_serialise:
                self.logger.info("Serialising pretrain data..")
                action = {**self.parallelism, **load_action}
                self.pretrain_file.write('{}%{}%{}%{}\n'.format(system_state.as_dict(),
                                                                action, system_reward.states_dict(), terminal))
                self.pretrain_file.flush()
            if self.log_rewards:
                action = {**self.parallelism, **load_action}
                self.logger.info("Writing reward to file..")
                self.rewards_file.write('{}%{}%{}%{}\n'.format(system_state.as_dict(),
                                                               action, system_reward.states_dict(), terminal))
                self.rewards_file.flush()

            # log latency and throughput
            system_reward = system_reward.states_dict()
            total_latency += system_reward['metrics']['latency']
            total_throughput += \
                (system_reward['metrics']['ack_count'] - prev_ack_count) / self.delay
            prev_ack_count = system_reward['metrics']['ack_count']

        self.logger.info('Total Latency: {}'.format(total_latency))
        self.logger.info('Total Throughput: {}'.format(total_throughput))
        rewards = np.asarray(rewards)
        self.logger.info('Average Reward: {}'.format(np.mean(rewards)))
        return rewards

    def start_topology(self):
        self.logger.info('Starting Topology')
        return_code = start_topology(self.cluster_role_env, self.path_to_jar,
                                     self.class_name, self.topology, load_dhalion=self.load_dhalion,
                                     config_path=self.config_path)
        if return_code:
            self.logger.error('Failed to submit topology')
            return
        self.logger.info('Submitted Topology Successfully')

    def run(self, test_only=False):
        # run the experiment by:
        # 0. Create the topology
        self.start_run = time.perf_counter()

        self.start_topology()
        # wait a bit for the topology to establish itself
        self.logger.info('Waiting for the topology to start before starting model')
        time.sleep(self.wait_time)
        self.system_model = HeronSystemModel(self.cluster, self.role, self.env,
                                             self.topology, self.parallelism,
                                             self.experiment_config['acceptable_misses'],
                                             failures=self.failures, delay=self.delay,
                                             wait_time=self.wait_time,
                                             config_path=self.experiment_config['config_path'],
                                             max_instances=self.experiment_config['max_instances'],
                                             max_over_instances=
                                             self.experiment_config['max_over_instances'])
        self._init_model_generator()
        self.waiting_on_system_time += (time.perf_counter() - self.start_run)
        if self.pretrain_serialise:
            self.pretrain_file = open(os.path.join(self.demo_dir, self.demo_file),
                                      'a')
        if self.log_rewards:
            self.rewards_file = open(os.path.join(self.results_dir, self.result_file), 'a+')

        # 1.a) Create Load using the load process
        # 1.b) Save this generated load
        # 2. Run a random agent (eventually) TODO
        # 3. Run a rule-based agent (eventually) TODO 
        # 4. Load a pretrained agent (Eventually) TODO
        # 5. Train the agent online against the load process
        if not test_only:
            self.logger.info('Starting Training')
            self.train(replay=False)
            end_run = time.perf_counter()
            self.total_time = end_run - self.start_run
            self.logger.info('Total Time Elapsed: {}'.format(self.total_time))
        # 6. Test the agent against the saved generated load
        #    using the saved generated load earlier.
        rewards = self.test()
        print(rewards)
        self.close()
