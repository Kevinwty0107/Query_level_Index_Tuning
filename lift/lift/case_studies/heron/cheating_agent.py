# This class makes many assumptions about the 
# nature of the task. However, it solves it 
# very well. 
class CheatingAgent(object):
    
    def __init__(self, agent_config):
        self.increase_amount = agent_config['increase_amount']
        self.low_spout_par = agent_config['lower_spout_par']
        self.upper_spout_par = agent_config['upper_spout_par']
        self.num_bolts = agent_config['num_bolts']
        # assume that we will start on the high spout parallelism
        self.prev_spout_par = self.low_spout_par
        self.runs = 0
        self.prev_action = 0
        self.changed_last = False
        self.change_back = False
    
    def act(self, states, independent=False, deterministic=False, wait=False):
        spout_par = states['spout_par']
        par = states['par']
        if spout_par == self.low_spout_par:
            if par == 3:
                action = 0
            elif par == 6:
                action = -3
            elif par == 8:
                action = -2
        elif spout_par == self.upper_spout_par:
            if par == 3:
                action = 3
            elif par == 6:
                action = 2
            elif par == 8:
                action = 0
        else:
            raise RuntimeError('Not a valid spout parallelism for this '
                    'demonstration agent')

        return {'par' : self.increase_amount + action}

    #def act(self, states, independent=False, deterministic=False, wait=False):
    #    # if haven't been run before then do not change it this round
    #    # cheat by flinging ourselves between two decent configs. 
    #    spout_par = states['spout_par']
    #    increased = spout_par == self.upper_spout_par and self.prev_spout_par == \
    #            self.low_spout_par 
    #    decreased = spout_par == self.low_spout_par and self.prev_spout_par == \
    #            self.upper_spout_par
    #    if increased or decreased:
    #        if wait:
    #            self.changed_last = True
    #            action = 0
    #        else:
    #            if increased:
    #                action = 3
    #            else:
    #                action = -3
    #    elif wait and self.changed_last:
    #        if spout_par == self.low_spout_par:
    #            action = -3
    #        else:
    #            action = 3
    #        self.change_back = True
    #    elif self.prev_action == 3:
    #        action = 2
    #    elif self.prev_action == -3:
    #        action = -2
    #    else:
    #        action = 0
    #    if self.runs % self.num_bolts == self.num_bolts - 1:
    #        self.prev_spout_par = spout_par
    #        self.prev_action = action
    #        if wait and self.change_back:
    #            self.changed_last = False
    #            self.change_back = False
    #    self.runs += 1        
    #    return {'par' : self.increase_amount + action}

    def observe(self, terminal=False, reward=0.00):
        # we are rule-based fool
        pass
