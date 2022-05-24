

class OptimisationInterface(object):
    """
    An abstract optimisation interface for baseline optimisations. This package
    and interface is meant as a bridge to third party optimisation libraries.

    Baseline optimizers are used to evaluate the difficulty of a task and serve as sanity
    checks for model assumptions.

    """

    def act(self, states, *args, **kwargs):
        """
        Returns an action for the current state. Note that the optimiser need not be
        stateful, i.e. it might not use state information.

        Args:
            states (dict): State information.
            *args: Any additional args required for the optimiser.
            **kwargs: Any additional kwargs required for the optimiser.
        """
        pass

    def observe(self, performance, *args, **kwargs):
        """
        Report performance to the optimiser.

        Args:
            performance (any): Single performance metric.
            *args: Any other args required to describe performance
            **kwargs: Any other kwargs required to describe performance
        """
        pass

    def run(self, *args, **kwargs):
        """
        Some optimisers do not expose an external API but instead require users to subclass
        a special run interface. In this case, call the subclassed interface from here.
        """
        pass

    def eval_best(self, config, label):
        """
        Evaluates final configuration, exports result with label.
        """
        pass
