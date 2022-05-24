import logging
import numpy as np
from lift.model.model_generator import ModelGenerator
from lift.model.action import Action
from lift.model.reward import Reward


class HeronModelGenerator(ModelGenerator):
    
    def __init__(self, constant_state, latency_normaliser,
            throughput_normaliser, reward_generator, experiment_config):
        super(HeronModelGenerator, self).__init__()
        # encodes everything about the system
        # that will not change.
        self.logger = logging.getLogger(__name__)
        self.constant_state = constant_state
        self.latency_normaliser = latency_normaliser
        self.throughput_normaliser = throughput_normaliser
        self.reward_generator = reward_generator
        self.prev_ack_count = 0
        self.delay = constant_state['delay']
        # the order of this list will be the god-given
        # convention we follow to turn agent actions
        # into system actions
        self.bolts = list(constant_state['bolts'])
        self.experiment_config = experiment_config

    def agent_to_system_action(self, actions, parallelism):
        # get the actual action state_value
        action = actions['par']
        # init system action
        system_action = dict() 
        # use indices of bolts to map between the array
        # we get from the agent to the dictionary
        # required by the system
        for index, component in enumerate(self.bolts):
            # add 1 to ensure non-zero parallelisms
            system_action[component] = parallelism[component] + action[index]
        return Action(system_action)
   
    # system reward is just the latency and throughput dict
    # Confusingly reward_obj is a TODO STATE TODO object 
    def system_to_agent_reward(self, reward_obj):
        reward = reward_obj.as_dict()
        # convert the reward using something like a EMA or normalised 
        # average into a single objective
        latency = np.sqrt(reward['metrics']['latency'])
        throughput = (reward['metrics']['ack_count'] - self.prev_ack_count) / \
                self.delay
        instances = sum(reward['par'])
        # normalise latency and throughput by using pre-collected 
        # data or some shizzle.
        latency = self.latency_normaliser.scale(latency)
        throughput = self.throughput_normaliser.scale(throughput)
        data = dict()
        data['latency'] = latency
        data['throughput'] = throughput
        data['instances'] = instances
        # combine in some way
        return Reward(self.reward_generator.reward(data))


