import numpy as np


def action_equals(correct_action, agent_action):
    """
    Compares actions of different types and layouts.

    Args:
        correct_action (Union[dict, list, float, int]): Correct action.
        agent_action (Union[dict, list, float, int]): Agent action.

    Returns:
        True if actions match exactly for all sub actions.
    """
    if isinstance(agent_action, dict):
        # Compare dict
        all_equal = True
        for name, action_value in agent_action.items():
            # Compare sub action by type -> list or numerical state_value
            correct_value = correct_action[name]
            if isinstance(action_value, list):
                all_equal = all_equal & all(x == y for x, y in zip(correct_value, action_value))
            if isinstance(action_value, np.ndarray):
                all_equal = all_equal & np.array_equal(correct_value, action_value)
            else:
                all_equal = all_equal & (correct_value == action_value)
        return all_equal
    elif isinstance(agent_action, list):
        # Compare list state_value-by-state_value.
        return all(x == y for x, y in zip(correct_action, agent_action))
    else:
        # Compare simple values.
        return correct_action == agent_action
