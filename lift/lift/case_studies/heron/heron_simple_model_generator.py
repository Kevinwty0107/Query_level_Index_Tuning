import numpy as np
from lift.model.state import State
from lift.case_studies.heron.heron_model_generator import HeronModelGenerator


class HeronSimpleModelGenerator(HeronModelGenerator):
    
    def __init__(self, constant_state, latency_normaliser,
            throughput_normaliser, reward_generator, experiment_config):
        # encodes everything about the system
        # that will not change.
        super(HeronSimpleModelGenerator, self).__init__(constant_state,
                latency_normaliser, throughput_normaliser, reward_generator,
                experiment_config)
        self.no_nodes = len(constant_state['adj'])
            
    def system_to_agent_state(self, message):
        if message is None:
            self.logger.info('Message returned was None')
            return None
        message = message.as_dict()
        agent_state = dict()
        agent_state['adj'] = self.constant_state['adj']
        agent_state['par'] = message['par']

        metrics = message['metrics']
        # merge the metrics into one 
        agent_metrics = np.zeros((self.no_nodes, len(metrics)))
        it = 0
        for metric, values in metrics.items():
            agent_metrics[:,  it] = values
            it = it + 1
        agent_state['metrics'] = agent_metrics
        return State(agent_state)
        
        
                
        
