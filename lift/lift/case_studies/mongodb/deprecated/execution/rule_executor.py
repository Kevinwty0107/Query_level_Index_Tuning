import time

from lift.case_studies.mongodb.deprecated.execution import Executor


class RuleExecutor(Executor):

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
        self.agent_config = agent_config

    def execute(self):
        states = self.get_observations()
        if len(states) == 0:
            return False

        executed = 0
        action_info = []

        self.logger.info('Embeddings in step = {}'.format(len(states)))
        for state in states:
            if time.time() > self.end:
                return

            # Include index information in state
            noop = False
            action = self.get_action_rule(states=states)
            # Action

            # self.logger.info(action)
            if self.no_op(action):
                self.logger.info('No-op true')
                noop = True

            # action_info.append(action_list)
            # Actually execute action in real system
            if not noop and executed < self.actions_per_interval:
                existed = self.act(action)
                if not existed:
                    executed += 1

        return True

    def get_action_rule(self, states):
        pass


