class ConstantAgent(object):
    
    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.action = agent_config['action']

    def act(self, states, independent=False, deterministic=False):
        return {'par' : self.action} 
        
    def observe(self, terminal=False, reward=0.00):
        # we are rule-based fool
        pass
