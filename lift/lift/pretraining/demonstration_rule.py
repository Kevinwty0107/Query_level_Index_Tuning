

class DemonstrationRule(object):


    """
    A demonstration rule is used to generate a demonstration.

    Demonstration rules can be context-free or contextual. A context-free
    rule can generate demonstrations sample-by-sample without consideration for prior decisions (which may be
    encoded as part of the state).

    A contextual demonstration attempts to generate demonstrations for
    a trajectory of arbitrary length.

    """
    def __init__(self, reward=0, margin=0):
        self.demo_reward = reward
        self.demo_margin = margin

    def generate_demonstration(self, states, **kwargs):
        """
        Generates one or multiple demonstrations.

        Args:
            states (any): Input states.

        Returns:
            any: System-actions
        """
        raise NotImplementedError

    def reward(self):
        """
        Returns a reward estimate for demo. Rewards can be used to rank demos with positive and negative
        examples.

        Returns:
            float: Optional reward.
        """
        return self.demo_reward

    def margin(self):
        """
        Returns confidence margin. Positive values indicate action is encouraged,
        negative values indicate action is discouraged.
        """
        return self.demo_margin
