from lift.case_studies.mongodb.deprecated.execution import Executor
from lift.case_studies.mongodb.deprecated.execution import RLExecutor


class NoOpExecutor(RLExecutor):

    def __init__(
        self,
        experiment_config,
        agent_config=None,
        states_spec=None,
        actions_spec=None,
        network_spec=None,
        model=None,
        end=None,
        serialize=False
    ):

        Executor.__init__(self, experiment_config, model, end, serialize)
        self.actions_spec = actions_spec

    def init_model(self):
        super(NoOpExecutor, self).init_model()
        self.agent = NoOpAgent(self.actions_spec)


class NoOpAgent(object):

    def __init__(self, actions_spec):
        self.no_op = dict()
        for name, action in actions_spec.items():
            # TODO note that this is not always no-op
            self.no_op[name] = 0

    def act(self, state):
        return self.no_op

    def observe(self, reward, terminal):
        pass

    def load_model(self, path=''):
        pass

    def store_model(self, path=''):
        pass
