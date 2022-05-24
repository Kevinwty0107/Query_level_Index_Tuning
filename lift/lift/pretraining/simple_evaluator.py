from lift.pretraining.evaluator import Evaluator
import numpy as np

from lift.util.pretrain_util import action_equals


class SimpleEvaluator(Evaluator):

    def __init__(self, agent, check_conflicts=True):
        """
        A simple evaluator just counts state and action exact matches to compute accuracy.

        Args:
            check_conflicts: Whether to memorize state and action pairs to identify
                potential state aliasing conflicts (multiple correct actions given for same state). This can happen
                if the state representation does not contain information necessary to distinguish two similar states.
        """
        super(SimpleEvaluator, self).__init__(agent=agent)
        # Accuracy per epoch.
        self.training_history = []
        self.check_conflicts = check_conflicts
        if check_conflicts:
            # Maps states to their demos
            self.state_memory = {}

    def evaluate(self, states, actions, **kwargs):
        data_points = float(len(states))
        correct = 0.0
        conflicts_found = 0

        for i in range(len(states)):
            if self.check_conflicts:
                state_str = str(states[i])
                if state_str not in self.state_memory:
                    self.state_memory[state_str] = actions[i]
                else:
                    # Test if the memoized action is same as current correct action, otherwise note conflict.
                    memoized_action = self.state_memory[state_str]
                    if not action_equals(correct_action=actions[i], agent_action=memoized_action):
                        conflicts_found += 1
                        self.logger.info("Conflict identified for state {}."
                                         " Memoized action {} but correct action is {}.".
                                         format(state_str, memoized_action, actions[i]))
            # This may be a dict with multiple values, list or a single state_value
            batched_state = np.asarray([states[i]])
            agent_action = self.agent.get_action(states=batched_state, use_exploration=False, apply_preprocessing=False)
            if action_equals(correct_action=actions[i], agent_action=agent_action):
                correct += 1.0

        accuracy = correct / data_points
        self.logger.info("Mean accuracy ({} data points): {}".format(data_points, accuracy))
        if self.check_conflicts:
            self.logger.info("Action conflicts encountered in training: {}".format(conflicts_found))
        self.training_history.append(accuracy)

    def export(self, result_dir, **kwargs):
        data = np.asarray(self.training_history)
        path = result_dir + '/pretrain_accuracy.txt'
        np.savetxt(path, data, delimiter=',')
