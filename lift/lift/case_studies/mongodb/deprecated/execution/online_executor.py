import tensorflow as tf
from tensorforce.agents import Agent

from lift.case_studies.mongodb.deprecated.execution import Executor
from lift.case_studies.mongodb.deprecated.execution import RLExecutor


class OnlineExecutor(RLExecutor):
    """
    Executes deep q learning from demonstration.
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
        self.network_spec = network_spec
        self.agent_config = agent_config

    def init_model(self):
        super(OnlineExecutor, self).init_model()
        tf.reset_default_graph()

        self.agent = Agent.from_spec(
            spec=self.agent_config,
            kwargs=dict(
                states_spec=self.states_spec,
                actions_spec=self.actions_spec,
                network_spec=self.network_spec
            )
        )