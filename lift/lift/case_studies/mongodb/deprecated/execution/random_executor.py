from tensorforce.agents import Agent

from lift.case_studies.mongodb.deprecated.execution import Executor
from lift.case_studies.mongodb.deprecated.execution import RLExecutor


class RandomExecutor(RLExecutor):
    """
    Executes a random action per time interval.
    """

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
        self.agent_config = agent_config

    def init_model(self):
        super(RandomExecutor, self).init_model()
        # Overwrite agent type in config
        self.agent_config['type'] = "random"
        self.agent = Agent.from_spec(
            spec=self.agent_config,
            kwargs=dict(
                states_spec=self.states_spec,
                actions_spec=self.actions_spec,
            )
        )
