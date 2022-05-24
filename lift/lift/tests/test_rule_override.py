import unittest
import os
import numpy as np
from rlgraph.agents import DQFDAgent
from rlgraph.spaces import BoolBox, FloatBox, IntBox, Dict
from rlgraph.tests.test_util import config_from_path, recursive_assert_almost_equal


class TestRuleOverride(unittest.TestCase):
    """
    How many updates does it take to learn to override a pretrained rule?
    """
    @staticmethod
    def current_reward(runtime=2.0, size=1.0):
        return -runtime - size

    @staticmethod
    def new_reward(runtime=2.0, size=1.0):
        x = [runtime, size]
        if sum(x) > 0:
            return np.exp(1 / sum(x))
        else:
            return 0


    @staticmethod
    def termwise_reward(runtime=2.0, size=1.0, regularizer=0.1):
        r = 0
        if runtime > 0:
            r += 1.0 / (runtime + regularizer)
        if size > 0:
            r += 1.0 / size
        return r

    def test_r(self):
        print(self.new_reward())

        runtimes = np.arange(0, 3, 0.001)
        size = np.arange(0.1, 3, 0.1)
        regularizers = [0.001, 0.01, 0.1]
        for regular in regularizers:
            for runtime, s in zip(runtimes, size):
                reward = self.termwise_reward(runtime, s, regular)
                print("Regularizer {} Runtime = {}, size = {}, reward = {}".format(regular, runtime, s, reward))

    def test_override(self):
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
            action_space=actions_space,
            store_last_q_table=True
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
        demo_rewards[0] = 0
        demo_rewards[1] = 0

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
        margins = [0.1, 0.1]

        # Fit demos.
        losses = []
        for _ in range(100):
            loss, _ = agent.update(batch, expert_margins=margins, apply_demo_loss_to_batch=True)
            losses.append(loss)
        print("Losses = ", losses)
        q_table = agent.last_q_table
        print("q_table = ", q_table)

        q_values = q_table["q_values"][0]

        # Compare q-values for best action.
        print("q value for best action = ", q_values[best_index])

        # Check margins:
        diffs_best = []
        for q_value in q_values:
            diffs_best.append(q_values[best_index] - q_value)

        print("Diffs between best q value and other q values: ", diffs_best)
        print("################## ################## ##################")
        # Now create a different action - non demo.
        new_action = [2, 2]
        demo_rewards = [self.new_reward(2.2, 3), self.new_reward(2.2, 3)]
        batch = dict(
            states=demo_states,
            actions=new_action,
            rewards=demo_rewards,
            next_states=demo_next_states,
            terminals=demo_terminals,
            importance_weights=np.ones_like(demo_rewards)
        )
        losses = []
        for _ in range(1000):
            loss, _ = agent.update(batch, apply_demo_loss_to_batch=False)
            losses.append(loss)
        print("Losses = ", losses)
        q_table = agent.last_q_table
        print("q_table = ", q_table)

        q_values = q_table["q_values"][0]

        # Compare q-values for best action.
        print("q value for demonstrated action = ", q_values[best_index])
        print("q value for new high reward action = ", q_values[2])