from lift.case_studies.heron.heron_agent_schema import HeronAgentSchema
class HeronFullStateSchema(HeronAgentSchema):


    def __init__(self, experiment_config):
        super(HeronFullStateSchema, self).__init__(experiment_config)
        self.max_instances = experiment_config['max_instances'] 


    def get_states_spec(self):
        # TODO add + 1 if using par
        state = dict(
                metrics=dict(type='float',continuous=True, 
                    shape=(len(self.metric_dicts), self.max_instances))
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
