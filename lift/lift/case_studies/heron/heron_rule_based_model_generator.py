from lift.model.state import State
from lift.case_studies.heron.heron_model_generator import HeronModelGenerator

class HeronRuleBasedModelGenerator(HeronModelGenerator):
    
    def __init__(self, constant_state, latency_normaliser, 
            throughput_normaliser, reward_generator, experiment_config):
        super(HeronRuleBasedModelGenerator, self).__init__(constant_state,
                latency_normaliser, throughput_normaliser, reward_generator,
                experiment_config)
        
    def system_to_agent_state(self, system_state_obj):
        system_state = system_state_obj.as_dict()
        agent_state = dict()
        agent_state['cpu'] = system_state['metrics']['cpu']
        agent_state['par'] = system_state['par']
        agent_state['spout_par'] = system_state['spout_par']
        agent_state['par'] = system_state['par']
        return State(agent_state)

    
