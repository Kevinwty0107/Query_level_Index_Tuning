import os
import json
import logging

import numpy as np
import pandas as pd

from lift.pretraining.data_source import DataSource
from lift.model.state import State
from lift.case_studies.heron import heron_model_generators, \
    heron_reward_generators
from lift.case_studies.heron.scaler import ObjectiveScaler


class PretrainDataSource(DataSource):

    def __init__(self, experiment_config):
        super(PretrainDataSource, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.spout = experiment_config['load_config']['component']
        self.test_file = os.path.join(experiment_config['demo_dir'],
                                      experiment_config['test_file'])
        self.experiment_config = experiment_config
        self.components = list(self.experiment_config['parallelism'].keys())
        self._init_model_generator()

    def _init_model_generator(self):
        df = pd.read_csv(self.experiment_config['reward_csv'])
        df['latency_sqrt'] = df['latency'].apply(np.sqrt)
        df = df.groupby(self.components, as_index=False).agg(
            {'latency_sqrt': ['mean', PretrainDataSource._stddev, 'count'],
             'throughput': ['mean', PretrainDataSource._stddev, 'count']})
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
        constant_state = dict()
        constant_state['delay'] = self.experiment_config['delay']
        constant_state['bolts'] = self.experiment_config['bolts']
        self.use_max = self.experiment_config['use_max']
        self.model_generator = heron_model_generators[ \
            self.experiment_config['converter']](
            constant_state, latency_scaler,
            throughput_scaler, reward_generator,
            self.experiment_config)

    def get_evaluation_data(self):
        return self.load_data(self.test_file, return_tuple=True)

    def load_rewards(self, save_file):
        rewards = []
        steps = self.load_csv(save_file, delimiter='%')
        for i in range(len(steps)):
            step = steps[i]
            reward_str = str(step[2]).replace("\'", "\"")
            reward_dict = json.loads(reward_str)
            if not isinstance(reward_dict, dict):
                print('Reward: {}, File: {}, Iteration: {}'.format(reward_dict,
                                                                   save_file, i))
            reward = State(reward_dict)
            for component in self.components:
                if component == self.spout:
                    continue
                # pass this through the converter
                agent_reward = self.model_generator.system_to_agent_reward(
                    reward)
                rewards.append(agent_reward)
        return rewards

    def load_data(self, save_file, return_tuple=False, concat=False):
        if concat and return_tuple:
            raise RuntimeError('Invalid Configuration Options')
        if not return_tuple:
            data = []
        else:
            states = []
            actions = []
            rewards = []
            terminals = []

        steps = self.load_csv(save_file, delimiter='%')
        for i in range(len(steps)):
            step = steps[i]
            # Replace single quotes:
            state_string = str(step[0]).replace("\'", "\"")
            state = State(json.loads(state_string))
            reward_string = str(step[2]).replace("\'", "\"")
            reward_dict = json.loads(reward_string)
            if not isinstance(reward_dict, dict):
                print('Reward: {}, File: {}, Iteration: {}'.format(reward_dict,
                                                                   save_file, i))
            reward = State(reward_dict)
            action = reward.as_dict()['par']
            prev_action = state.as_dict()['par']
            if concat:
                states = []
                actions = []
            for component in self.components:
                if component == self.spout:
                    continue
                index = state.as_dict()['name_to_index'][component]
                component_state = PretrainDataSource.extract_component(state, index, self.spout)

                # pass this through the converter
                prev_component_action = prev_action[index]
                component_action = action[index]
                agent_state = self.model_generator.system_to_agent_state(
                    component_state, use_max=self.use_max)
                agent_action = self.model_generator.system_to_agent_action(
                    prev_component_action, component_action)
                agent_reward = self.model_generator.system_to_agent_reward(
                    reward)
                self.logger.debug('Tuple: {}%{}%{}%{}'.format(agent_state,
                                                              agent_action,
                                                              agent_reward.get_value(),
                                                              False))
                if not return_tuple:
                    data.append(dict(
                        states=agent_state.as_dict(),
                        actions=agent_action,
                        terminal=False,
                        internals={},
                        reward=agent_reward.get_value()))
                elif concat:
                    states.append(agent_state.as_dict())
                    actions.append(agent_action)
                else:
                    states.append(agent_state)
                    actions.append(agent_action)
                    rewards.append(agent_reward)
                    terminals.append(False)
            if concat:
                data.append(dict(
                    states=states,
                    actions=actions,
                    terminal=False,
                    internals={},
                    reward=agent_reward.get_value()))
        if not return_tuple:
            ret = data
        else:
            ret = (states, actions, rewards, terminals)

        return ret

    def load_agent_actions(self, save_file):
        data = dict()
        steps = self.load_csv(save_file, delimiter='%')
        for step in steps:
            reward_string = str(step[2]).replace("\'", "\"")
            rewards = json.loads(reward_string)
            action = rewards['par']
            state_string = str(step[0]).replace("\'", "\"")
            state = json.loads(state_string)
            # prev_action = state['par']
            for component in self.components:
                if component == self.spout:
                    continue
                index = state['name_to_index'][component]
                component_action = action[index]
                # prev_component_action = prev_action[index]
                # agent_action = self.converter.system_to_agent_action(
                #    prev_component_action, component_action)
                if component in data:
                    data[component].append(component_action)
                else:
                    data[component] = [component_action]
        return data

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

    @staticmethod
    def _stddev(x):
        return x.std() / np.sqrt(x.count())