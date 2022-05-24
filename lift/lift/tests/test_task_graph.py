import unittest
from copy import copy
import numpy as np

from rlgraph.spaces import IntBox, FloatBox
from rlgraph.tests import config_from_path

from lift.rl_model.task import Task
from lift.rl_model.task_graph import TaskGraph


class TestTaskGraph(unittest.TestCase):

    def test_hierarchical_graph(self):
        g = TaskGraph()

        input_states = FloatBox(shape=(10,))
        agent_config = config_from_path("configs/dqfd_agent_for_cartpole.json")
        actions_spec = IntBox(low=0, high=5)

        network_spec = [dict(type='dense', units=32, activation='relu', scope="dense_1")]
        agent_config["network_spec"] = network_spec

        # Postprocessing hook.
        def post_act_hook(s):
            return np.expand_dims(s, -1)

        top_task = Task(agent_config, input_states, actions_spec, name="top_task", post_processing_fn=post_act_hook)

        dependent_config = copy(agent_config)

        # Take in output action, convert to float.
        dependent_config["network_spec"] = [
            dict(type="convert_type", to_dtype="float", scope="convert_type"),
            dict(type='dense', units=32, activation='relu', scope="dense_1")
        ]
        intermediate_states = IntBox(shape=(1,), low=0, high=5)

        dependent_task = Task(dependent_config, intermediate_states, actions_spec, "dependent_task")
        top_task.add_subtask(dependent_task)
        g.add_task(top_task)

        self.assertIsNotNone(g.get_task("top_task"))
        self.assertIsNotNone(g.get_task("dependent_task"))

        # Now try acting through the task graph.
        action = g.act_task(top_task.name, input_states.sample(), propagate=True)
        # Dict of task names to actions.
        print(action)

        # Only act on top task.
        action = g.act_task(top_task.name, input_states.sample(), propagate=False)
        # Only one action, is unpacked to its type (int).
        print(action)


