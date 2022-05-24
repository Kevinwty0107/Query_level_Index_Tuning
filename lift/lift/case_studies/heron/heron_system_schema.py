from lift.rl_model.schema import Schema

class HeronSystemSchema(Schema):

    def __init__(self, components, metrics, delay=60):
        Schema.__init__(self)
        self.no_nodes = len(components)
        self.no_metrics = len(metrics)
        self.delay = delay
        self.components = components
        self.metrics = metrics
    
    def get_states_spec(self):
        """
        Return a dictionary indicating
        the states available. Of the 
        format dict[nameOfStateComponent] -> dict of type and shape
        or can not have the name of the space component. Must be 
        encoded so as to be constant (i.e. not altered by the 
        actions)
        Components:
            par: Parallelism dictionary
            metrics: metrics dictionary
        """
        
        ms = dict()
        for metric in self.metrics:
            if metric == 'latency' or metric == 'ack_count':
                ms[metric] = dict(type='float', shape=())
            else:
                ms[metric] = dict(type='float', shape=(self.no_nodes,))
            
        return dict(
            par = dict(type='int', shape=(self.no_nodes,)),
            metrics = ms,
        )

    def get_actions_spec(self):
        """
        Return a dictionary containing
        the actions available. Similar to the
        states dictionary, but with the actions. Must
        be encoded so as to be constant (i.e. not
        altered by the actions).
        Components:
            par: Parallelism matrix (no_nodes,)
        """
        return dict(
            par = dict(type='int', no_actions=self.no_nodes)
        )
    
    def get_system_spec(self):
        n_to_i = dict()
        for component in self.components:
            n_to_i[component] = dict(type='int', shape=())
        return dict(
            adj = dict(type='bool', shape=(self.no_nodes, self.no_nodes)),
            delay = dict(type='int', shape=()),
            name_to_index = n_to_i
        )

    def __str__(self):
        return 'TwitterHeronSystemSpec'

