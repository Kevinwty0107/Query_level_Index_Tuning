from tensorforce.agents import Agent
from lift.case_studies.mongodb.deprecated.execution import Executor
from lift.case_studies.mongodb.deprecated.execution import RLExecutor


class PretrainExecutor(RLExecutor):

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
        self.states_spec = states_spec
        self.network_spec = network_spec
        self.agent_config = agent_config

    def init_model(self):
        super(PretrainExecutor, self).init_model()
        assert self.agent_config['type'] in ['dqfd']
        self.agent = Agent.from_spec(
                spec=self.agent_config,
                kwargs=dict(
                    states_spec=self.states_spec,
                    actions_spec=self.actions_spec,
                    network_spec=self.network_spec
                )
            )

    def import_train_data(self, data):
        self.agent.set_demonstrations(data)

    def pretrain(self, steps):
        """
        Pretrains model
        """
        self.agent.run(steps)


