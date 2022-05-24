from lift.rl_model.schema import Schema

class HeronSimpleAgentSchema(Schema):

    def __init__(self, experiment_config):
        super(HeronSimpleAgentSchema, self).__init__()
        self.no_nodes = len(self.experiment_config['parallelism'].keys())
        self.no_metrics = len(self.experiment_config['state_metrics'])
        self.components = list(self.experiment_config['parallelism'].keys())
        self.num_actions = self.experiment_config['max_instances'] - \
                len(self.components) + 1

    def get_states_spec(self):

        return dict(
                par=dict(type='int',continuous=False, shape=(self.no_nodes,)),
                adj=dict(type='bool', continuous=False, 
                    shape=(self.no_nodes, self.no_nodes)),
                metrics = dict(type='float', 
                    shape=(self.no_nodes,self.no_metrics)),)

    def get_actions_spec(self):
        return dict(
                par=dict(type='int', continuous=False, 
                    shape=(self.no_nodes - 1), num_actions=self.num_actions),)

    def get_system_spec(self):
        n_to_i = dict()
        for component in self.components:
            n_to_i[component] = dict(type='int', shape=())
        return dict(
            adj = dict(type='bool', shape=(self.no_nodes, self.no_nodes)),
            delay = dict(type='int', shape=()),
            name_to_index = n_to_i
        )


