from rlgraph.agents import DQNAgent
from rlgraph.agents import Agent


class Task(object):
    """
    Represents a single task in a hierarchical task graph.
    """
    def __init__(self, agent_config, state_space, action_space, name="", post_processing_fn=None):
        self.sub_tasks = {}
        self.agent = DQNAgent.from_spec(agent_config, state_space=state_space, action_space=action_space)
        #self.agent = Agent.from_spec(agent_config, state_space=state_space, action_space=action_space)
        self.name = name

        # Hook to post-process.
        self.post_processing_fn = post_processing_fn

    def add_subtask(self, task):
        if task.name in self.sub_tasks:
            raise ValueError("Task {} already exists in subtasks.".format(task.name))
        self.sub_tasks[task.name] = task

    def act(self, states, use_exploration=True, apply_preprocessing=True, time_percentage=None, propagate=True):
        # List of actions returned by tasks and sub-tasks.
        task_output = self.agent.get_action(states=states, use_exploration=use_exploration,
                                            apply_preprocessing=apply_preprocessing, time_percentage=time_percentage)
        actions = {self.name: task_output}

        # Propagate to sub-tasks, process intermediate result.
        if propagate is True:
            if self.post_processing_fn is not None:
                task_output = self.post_processing_fn(task_output)

            # Same output could be used by multiple sub-tasks.
            for name, task in self.sub_tasks.items():
                actions[name] = task.act(task_output)

        # Unpack.
        if isinstance(actions, dict) and len(actions) == 1:
            return list(actions.values())[0]
        else:
            return actions

    def observe(self, preprocessed_states, actions, internals, rewards, terminals, next_states=None):
        self.agent.observe(preprocessed_states=preprocessed_states, actions=actions,
                           internals=internals, rewards=rewards, next_states=next_states, terminals=terminals)

    def update(self, batch, *args):
        return self.agent.update(batch, *args)

    def unwrap(self):
        """
        Returns underlying task model (e.g. agent)/
        """
        return self.agent

    def reset(self):
        self.agent.reset()
