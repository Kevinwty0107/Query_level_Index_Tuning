import logging


class Evaluator(object):
    """
    Evaluates states and action pairs, e.g. between epochs or after training.
    """
    def __init__(self, agent):
        """
        Args:
            agent (any): An object implementing a get_action method to make predictions ofr states.
        """
        self.logger = logging.getLogger(__name__)
        self.agent = agent

    def evaluate(self, states, actions, **kwargs):
        """
        Evaluates a set of states with regard to how well they predict a set of actions.

        Args:
            states: List of states
            actions: List of actions
            kwargs: Additional evaluation args.

        Returns:
            Eval metrics, e.g. accuracy.
        """
        raise NotImplementedError

    def export(self, result_dir, **kwargs):
        """
        Exports training history.

        Args:
            result_dir (str): Directory to export to.
            **kwargs: Additional export info, e.g. labels.

        """
        raise NotImplementedError
