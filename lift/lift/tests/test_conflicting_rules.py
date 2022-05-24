import unittest
import os
import numpy as np
from lift.case_studies.common.index_net import build_index_net
from rlgraph.agents import DQFDAgent
from rlgraph.spaces import BoolBox, FloatBox, IntBox, Dict
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestConflictingRules(unittest.TestCase):

    def test_positive_only(self):
        # Tests if dqfd can fit a set of states to a set of actions.
        vocab_size = 32
        embed_dim = 16
        # ID/state space.
        state_space = IntBox(vocab_size, shape=(10,))
        # Container action space.
        num_outputs = 5
        actions_space = IntBox(
            low=0,
            high=num_outputs
        )

        agent_config = config_from_path("configs/agent_config.json", root=os.getcwd())
        agent_config["network_spec"] = [
            dict(type="embedding", embed_dim=embed_dim, vocab_size=vocab_size),
            dict(type="reshape", flatten=True),
            dict(type="dense", units=embed_dim, activation="relu", scope="dense_1")
        ]
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=state_space,
            action_space=actions_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)

        # Create a set of demos.
        demo_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        # Same state.
        demo_states[1] = demo_states[0]
        demo_actions = actions_space.with_batch_rank().sample(2)

        # Create a good example
        best_index = 0
        demo_actions[0] = best_index
        demo_actions[1] = best_index

        demo_rewards = rewards.sample(2, fill_value=.0)
        # One action has positive reward, one negative
        demo_rewards[0] = 1
        demo_rewards[1] = 1

        demo_next_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        demo_terminals = terminals.sample(2, fill_value=False)

        # Insert.
        batch = dict(
            states=demo_states,
            actions=demo_actions,
            rewards=demo_rewards,
            next_states=demo_next_states,
            terminals=demo_terminals,
            importance_weights=np.ones_like(demo_rewards)
        )

        # Positive margin for good example, negative for negative example.
        margins = [1.0, 1.0]

        # Fit demos.
        for _ in range(1000):
            agent.update(batch, expert_margins=margins, apply_demo_loss_to_batch=True)
        # Q values api deprecated in rlgraph
        # q_table = agent.last_q_table
        # print("q_table = ", q_table)
        #
        # q_values = q_table["q_values"][0]
        #
        # # Compare q-values for best action.
        # print("q value for best action = ", q_values[best_index])
        #
        # # Check margins:
        # diffs_best = []
        # for q_value in q_values:
        #     diffs_best.append(q_values[best_index] - q_value)
        #
        # print("Diffs between best q value and other q values: ", diffs_best)

    def test_conflicting_demos(self):
        # Tests if dqfd can fit a set of states to a set of actions.
        vocab_size = 100
        embed_dim = 32
        # ID/state space.
        state_space = IntBox(vocab_size, shape=(20,))
        # Container action space.
        num_outputs = 5
        actions_space = IntBox(
            low=0,
            high=num_outputs
        )

        agent_config = config_from_path("configs/agent_config.json", root=os.getcwd())
        agent_config["network_spec"] = [
            dict(type="embedding", embed_dim=embed_dim, vocab_size=vocab_size),
            dict(type="reshape", flatten=True),
            dict(type="dense", units=embed_dim, activation="relu", scope="dense_1")
        ]
        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=state_space,
            action_space=actions_space,
            store_last_q_table=True
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)

        # Create a good and bad example.
        worst_index = 4
        demo_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        # Same state.
        demo_states[1] = demo_states[0]
        demo_actions = actions_space.with_batch_rank().sample(2)

        # Create a good example
        best_index = 0
        demo_actions[0] = best_index
        demo_actions[1] = worst_index

        demo_rewards = rewards.sample(2, fill_value=.0)
        # One action has positive reward, one negative
        demo_rewards[0] = 1
        demo_rewards[1] = -1

        demo_next_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        demo_terminals = terminals.sample(2, fill_value=False)

        # Insert.
        batch = dict(
            states=demo_states,
            actions=demo_actions,
            rewards=demo_rewards,
            next_states=demo_next_states,
            terminals=demo_terminals,
            importance_weights=np.ones_like(demo_rewards)
        )

        margins = [1.0, -2.0]

        # Fit demos.
        for _ in range(1000):
            # Alternate between positive and negative batches
            agent.update(batch, expert_margins=margins, apply_demo_loss_to_batch=True)

        q_table = agent.last_q_table
        print("q_table = ", q_table)

        q_values = q_table["q_values"][0]

        # Compare q-values for best action.
        print("q value for best action = ", q_values[best_index])
        print("q value for worst action = ", q_values[worst_index])

        # Check margins:
        diffs_best = []
        diffs_worst = []
        for q_value in q_values:
            diffs_best.append(q_values[best_index] - q_value)
            diffs_worst.append(q_values[worst_index] - q_value)

        print("Diffs between best q value and other q values: ", diffs_best)
        print("Diffs between worst q value and other q values: ", diffs_worst)

    def test_with_index_net(self):
        # Tests if dqfd can fit a set of states to a set of actions.
        vocab_size = 32
        embed_dim = 16
        # ID/state space.
        state_space = Dict(
            sequence=IntBox(vocab_size, shape=(10,)),
            selectivity=FloatBox(shape=(10,)),
            add_batch_rank=True
        )

        # Container action space.
        num_outputs = 5
        actions_space = IntBox(
            low=0,
            high=num_outputs
        )

        agent_config = config_from_path("configs/agent_config.json", root=os.getcwd())
        # state_mode, states_spec, embed_dim, vocab_size, layer_size
        agent_config["network_spec"] = build_index_net(
            state_mode='index_net',
            states_spec=state_space,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            layer_size=embed_dim)

        agent = DQFDAgent.from_spec(
            agent_config,
            state_space=state_space,
            action_space=actions_space
        )
        terminals = BoolBox(add_batch_rank=True)
        rewards = FloatBox(add_batch_rank=True)

        # Create a set of demos.
        demo_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        demo_actions = actions_space.with_batch_rank().sample(2)
        demo_rewards = rewards.sample(2, fill_value=.0)
        demo_next_states = agent.preprocessed_state_space.with_batch_rank().sample(2)
        demo_terminals = terminals.sample(2, fill_value=False)

        print(demo_states)
        # Test act.
        print(agent.get_action(agent.preprocessed_state_space.with_batch_rank().sample()))
        print(agent.get_action(agent.preprocessed_state_space.with_batch_rank().sample(5)))
        # Insert.
        batch = dict(
            states=demo_states,
            actions=demo_actions,
            rewards=demo_rewards,
            next_states=demo_next_states,
            terminals=demo_terminals,
            importance_weights=np.ones_like(demo_rewards)
        )
        # Positive margin for good example, negative for negative example.
        margins = [1.0, 1.0]

        # Fit demos.
        for _ in range(10):
            agent.update(batch, expert_margins=margins, apply_demo_loss_to_batch=True)