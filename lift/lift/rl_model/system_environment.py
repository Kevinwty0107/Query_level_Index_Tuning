
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class SystemEnvironment(object):
    """
    A system environment describes the interface to a controlled system. A system model must typically
    implement operations to continuously retrieve new system states and execute changes to the system.
    """

    def observe_system(self, batch_size=0):
        """
        Observe new system states. This is for the case that the system does not
        directly return performance from act(), but rather performance is observed
        through some other mechanism, e.g. a log or api call somewhere.

        Args:
            batch_size (int): Optional batch size if results need to be limited by most recent
                states, e.g. most recent log entries.

        Returns:
            Iterable of state objects.
        """
        raise NotImplementedError

    def act(self, action):
        """
        Executes action in system. Can return performance depending
        on system semantics.

        :param action: RL agent output.
        """
        raise NotImplementedError

    def system_status(self):
        """
        Check system status for changes made to the system.
        Should retrieve current configuration.

        Returns:
             Current configuration
        """
        raise NotImplementedError

    def is_noop(self, action):
        """
        It can be convenient to define specifically which combinations, in case of multiple actions,
        are viewed as no-op actions. This can be used to prevent actions from being taken that are
        viewed as detrimental to the system, e.g. illegal or non-sensical actions.

        Returns:
            True if action is no-op.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset system in case of repeated experiments.
        """
        raise NotImplementedError
