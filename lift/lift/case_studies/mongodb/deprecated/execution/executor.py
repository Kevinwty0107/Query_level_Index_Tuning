from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import logging

from lift.case_studies.mongodb.deprecated.metrics import metrics_helpers


class Executor(object):
    """
    Executes the implementation of an MDP on a particular reinforcement learning agent.
    Collects runtime metrics related to actions executed, rewards seen etc.
    """

    def __init__(self, experiment_config, model, end, serialize):
        self.model = model

        self.logger = logging.getLogger(__name__)

        self.metrics_helper = metrics_helpers[experiment_config['metrics']]()
        self.agent = None
        self.end = end
        self.actions_per_interval = experiment_config['max_actions_per_interval']
        self.serialize = serialize

    def init_model(self):
        self.model.init_model()

    def execute(self):
        """
        Executes a logical step of the control model, which can correspond to multiple (or no) steps
        for the underlying reinforcement learning agent.
        :return:
        """
        raise NotImplementedError

    def no_op(self, action):
        return self.model.is_noop(action)

    def act(self, action):
        result = self.model.act(action)
        self.metrics_helper.record_result(result)

        return result['existed']

    def get_observations(self):
        parsed_log_entries = self.model.observe_system()
        self.logger.info('Observations in batch = {}'.format(len(parsed_log_entries)))

        if len(parsed_log_entries) > 0:
            if self.serialize:
                self.metrics_helper.serialize_observations(states=parsed_log_entries)

            # This maps a number of observations to an embedding, e.g. by counting unique queries
            # in the obervations and generating a single state per unique event.

            # TODO introduce flag if this is not desired?
            parsed_states, batch_reward = self.model.generate_state_batch(parsed_log_entries)
            self.metrics_helper.record_observations(
                observations=parsed_states,
                kwargs=dict(batch_reward=batch_reward)
            )

            return parsed_states
        else:
            return parsed_log_entries

    def export(self, path='', kwargs=None):
        self.metrics_helper.export_results(path, kwargs)