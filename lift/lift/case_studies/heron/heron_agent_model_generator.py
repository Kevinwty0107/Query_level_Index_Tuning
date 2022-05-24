import numpy as np
from lift.case_studies.heron.heron_model_generator import HeronModelGenerator
from lift.model.state import State


class HeronAgentModelGenerator(HeronModelGenerator):

    def __init__(self, constant_state, latency_normaliser,
                 throughput_normaliser, reward_generator, experiment_config):
        super(HeronAgentModelGenerator, self).__init__(constant_state,
                                                       latency_normaliser, throughput_normaliser, reward_generator,
                                                       experiment_config)
        self.max_decrease = self.experiment_config['max_decrease']
        self._init_metric_to_index()
        self.state_to_id = dict()
        self.id_counter = 0

    def _init_metric_to_index(self):
        self.metric_to_index = dict()
        for metric, spec in self.experiment_config['metric_dicts'].items():
            self.metric_to_index[metric] = spec['index']

    def system_to_agent_state(self, message, use_max=False):
        # get the metrics
        metrics = message.as_dict()['metrics']
        agent_state = dict()
        # Add + 1 if using par
        agent_state['metrics'] = np.zeros((len(metrics.keys()),))
        for metric, values in metrics.items():
            # compute the mean ignoring zeros
            if not use_max:
                total = np.sum(values)
                count = np.count_nonzero(values)
                if count != 0:
                    agent_state['metrics'][self.metric_to_index[metric]] = \
                        total / count
                else:
                    agent_state['metrics'][self.metric_to_index[metric]] = 0.0
            else:
                agent_state['metrics'][self.metric_to_index[metric]] = \
                    max(values)

        # put the parallelism on the end
        # actually don't do this bc want agent state to 
        # be independent of par
        # agent_state['metrics'][len(metrics.keys())] = \
        #        message.states_dict()['par']
        return State(agent_state)

    def system_to_agent_action(self, prev_component_action, component_action):
        return {'par': component_action - prev_component_action - \
                       self.max_decrease}
