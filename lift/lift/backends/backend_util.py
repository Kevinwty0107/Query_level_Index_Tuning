from lift import BACKEND

"""Various utilities to convert configs and instantiate agents for rlgraph/tensorforce"""


def get_agent_from_spec(config, states, actions, network_spec=None):
    """
    Generates an agent from a config dict, states and actions depending on the
    backend.
    """
    # TODO: Move all spaces to rlgraph -> then map backwards to tensorforce if needed

    if BACKEND == "tensorforce":
        from tensorforce.agents import Agent
        import tensorflow as tf
        tf.reset_default_graph()

        return Agent.from_spec(
            spec=config,
            kwargs=dict(
                states_spec=states,
                actions_spec=actions,
                network_spec=network_spec
            )
        )
    elif BACKEND == "rlgraph":
        from rlgraph.agents import Agent
        config["network_spec"] = network_spec
        actions = convert_actions(actions)
        states = convert_states(states)
        return Agent.from_spec(
            config,
            state_space=states,
            action_space=actions
        )


def set_learning_rate(config, learning_rate):
    """
    Updates learning rate backend-dependent.

    Args:
        config (dict): Agent config.

    Returns:
        dict: Updated agent config.
    """
    if BACKEND == "tensorforce":
        config['optimizer']['learning_rate'] = learning_rate
    elif BACKEND == "rlgraph":
        config['optimizer_spec']['learning_rate'] = learning_rate


def convert_actions(actions, to="rlgraph"):
    """
    Converts an actions spec.

    Args:
        actions (dict): Actions spec
        to (str): Backend to.
    """
    from rlgraph.spaces import Dict, IntBox
    spec = {}
    for name, action_dict in actions.items():
        spec[name] = IntBox(action_dict["num_actions"])

    return Dict.from_spec(spec)


def convert_states(states, to="rlgraph"):
    """
    Converts an actions spec.

    Args:
        states (dict): States spec
        to (str): Backend to.
    """
    from rlgraph.spaces import Dict, IntBox
    spec = {}
    for name, states in states.items():
        # N.b. high shouldnt matter for stsates because we never sample states.
        spec[name] = IntBox(low=0, high=100, shape=states["shape"])

    return Dict.from_spec(spec)


def convert_to_backend(network_spec, to="rlgraph"):
    """
    Converts a network spec from tensorforce to rlgraph. Only meant for temporary conversion
    before final move to rlgraph.

    Args:
        network_spec (list): List of layers
        to (str): Backend to convert to.

    Returns:
        list: Update spec.
    """
    index = 0
    out_list = []
    if to == "rlgraph":
        for layer in network_spec:
            layer_type = layer["type"]
            if layer_type == "flatten":
                out_list.append({
                        "type": "reshape",
                        "flatten": True
                    })
            elif layer_type == "embedding":
                out_list.append({
                    "type": "embedding",
                    "embed_dim": layer["size"],
                    "vocab_size": layer["indices"]
                })
            else:
                out_list.append({
                    "type": layer_type,
                    "units": layer["size"],
                    "activation": "relu",
                    "scope": "{}_{}".format(layer_type, index)
                })
                index += 1
    else:
        raise ValueError("Error: Cannot convert.")

