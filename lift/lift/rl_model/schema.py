

class Schema(object):
    """
    The schema helps managing shapes of various tensors to avoid
    one-off errors and incompatibilities.

    Models, data sources, and controllers use the schema to look-up data layouts from a single
    source of truth.
    """

    def get_actions_spec(self):
        """
        Describes actions as a dict mapping action names to action spaces.

        Returns:
            Actions dict.
        """
        raise NotImplementedError

    def get_states_spec(self):
        """
        Describes states as a dict mapping state name to state space.

        Returns:
            States dict.
        """
        raise NotImplementedError

    def get_system_spec(self):
        """
        Returns spec describing the controlled system, e.g. physical or logical data layout,
        additional shapes, etc.
        """
        raise NotImplementedError
