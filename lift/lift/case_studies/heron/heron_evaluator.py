from lift.pretraining.evaluator import Evaluator
import numpy as np

from lift.util.pretrain_util import action_equals


class HeronEvaluator(Evaluator):

    def __init__(self, agent, check_conflicts=True):
        """
        A simple evaluator just counts state and action exact matches to compute accuracy.

        :param agent:
        :param check_conflicts: Whether to memorize state and action pairs to identify
            potential state aliasing conflicts (multiple correct actions given for same state). This can happen
            if the state representation does not contain information necessary to distinguish two similar states.
        """
        super(HeronEvaluator, self).__init__(agent=agent)
        # Accuracy per epoch.
        self.training_history = []
        self.check_conflicts = check_conflicts
        self.agent_actions = []
         
        if check_conflicts:
            # Maps states to their demos
            self.state_memory = dict()

    def evaluate(self, states, actions, **kwargs):
        self.true_actions = actions
        data_points = float(len(states))
        correct = 0.0
        if self.check_conflicts:
            conflicts_found = 0
        self.agent_actions = []
        for i in range(len(states)):
            if self.check_conflicts:
                state_str = str(states[i])
                if state_str not in self.state_memory:
                    self.state_memory[state_str] = (i, actions[i])
                else:
                    # Test if the memoized action is same as current correct action, otherwise note conflict.
                    memoized_action = self.state_memory[state_str][1]
                    if not action_equals(correct_action=actions[i], agent_action=memoized_action):
                        conflicts_found += 1
                        self.logger.debug("Conflict identified on iteration {}"
                                "with iteration {}".format(i, 
                                    self.state_memory[state_str][0]))
                        self.logger.debug("Conflict identified for state {}."
                                         " Memoized action {} but correct action is {}.".
                                         format(state_str, memoized_action, actions[i]))
            # This may be a dict with multiple values, list or a single state_value
            agent_action = self.agent.act(states=states[i], independent=True, deterministic=True)
            self.agent_actions.append(agent_action)
            if action_equals(correct_action=actions[i], agent_action=agent_action):
                correct += 1.0

        accuracy = correct / data_points
        self.logger.info("Accuracy over {} data points: {}".format(data_points, accuracy))
        if self.check_conflicts:
            self.logger.info("Action conflicts encountered in training: {}".format(conflicts_found))
        self.training_history.append(accuracy)
        return accuracy

    def export(self, result_dir, **kwargs):
        data = np.asarray(self.training_history)
        path = result_dir + '/pretrain_accuracy.txt'
        np.savetxt(path, data, delimiter=',')
