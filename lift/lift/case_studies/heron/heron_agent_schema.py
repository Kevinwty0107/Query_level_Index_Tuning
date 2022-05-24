from lift.rl_model.schema import Schema

class HeronAgentSchema(Schema):


    def __init__(self, experiment_config):
        super(HeronAgentSchema, self).__init__()
        self.metric_dicts = experiment_config['metric_dicts']
        self.max_reduction = experiment_config['max_decrease']
        self.max_increase = experiment_config['max_increase']
    
    def get_states_spec(self):
        # TODO add + 1 if using par
        state = dict(
                metrics=dict(type='float',continuous=True, 
                    shape=(len(self.metric_dicts),))
        )
        return state

    def get_actions_spec(self):
        num_actions = self.max_increase - self.max_reduction + 1
        return dict(
                par=dict(type='int', continuous=False, num_actions=num_actions,
                    min=self.max_reduction, max=self.max_increase)
        )

    def get_system_spec(self):
        pass
