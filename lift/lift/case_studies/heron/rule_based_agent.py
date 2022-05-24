

class RuleBasedAgent(object):
    
    def __init__(self, agent_config):
        self.lower_bound = agent_config['lower_bound']
        self.higher_bound = agent_config['higher_bound']
        assert self.lower_bound < self.higher_bound 
        self.increase_amount = agent_config['increase_amount']
        self.max_par = agent_config['max_par']

    def act(self, states, independent=False, deterministic=False):
        capacities = states['cpu']

        # if they are all above the higher threshold 
        increase = True
        decrease = True
        for capacity in capacities:
            if capacity == 0.0:
                continue
            if capacity < self.higher_bound:
                increase = False
            if capacity > self.lower_bound:
                decrease = False
        if increase and decrease:
            # must have all been 0
            increase = False
            decrease = False
        action = 0
        if increase:
            # choose to increase the parallelism
            action = self.increase_amount
        if decrease:
            action = -self.increase_amount
        return {'par': min(self.increase_amount + action, self.max_par)}

    def observe(self, terminal=False, reward=0.00):
        # we are rule-based fool
        pass
