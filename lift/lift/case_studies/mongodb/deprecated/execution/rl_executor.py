import time

from lift.case_studies.mongodb.deprecated.execution import Executor


class RLExecutor(Executor):

    def execute(self):
        states = self.get_observations()
        if len(states) == 0:
            return False

        executed = 0
        self.logger.info('Embeddings in step = {}'.format(len(states)))
        for state in states:
            if time.time() > self.end:
                return

            # Include index information in state
            noop = False

            # Named action dict
            # self.logger.info(embedding.state)
            action = self.agent.get_action(dict(state=state.get_value()))
            meta_data = state.get_meta_data()
            self.logger.info('action ='.format(action))
            self.agent.observe(meta_data['reward'], False)

            self.metrics_helper.record_result(
                dict(
                    action_info=sorted(list(action.values()))
                )
            )
            # self.logger.info(action)
            if self.no_op(action):
                self.logger.info('No-op true')
                noop = True

            # Actually execute action in real system
            if not noop and executed < self.actions_per_interval:
                existed = self.act(action)
                if not existed:
                    executed += 1

        return True

    def restore_model(self, path=''):
        """
        Export trained TensorForce model.

        :param path: Model path
        :return:
        """
        # File for specific file, directory for directory
        self.agent.restore_model(file=path)

    def store_model(self, path=''):
        """
        Export trained TensorForce model.

        :param path: Model path
        :return:
        """
        self.agent.save_model(directory=path, append_timestep=True)