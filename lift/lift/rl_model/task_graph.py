import logging


class TaskGraph(object):
    """
    Represents a hierarchically organised collection of tasks.
    """

    def __init__(self):
        # Flat map of all sub-tasks.
        self.logger = logging.getLogger(__name__)
        self.tasks = {}

    def add_task(self, task):
        self.logger.info("Adding task {} to task-graph.".format(task.name))
        self.tasks[task.name] = task

        # Recursively add sub-tasks.
        to_add = []
        self.search_sub_tasks(task, to_add)

        for sub_task in to_add:
            assert task not in self.tasks, "Task {} is already in tasks.".format(sub_task.name)
            self.logger.info("Found sub-task {} for task {}, adding to task graph.".format(sub_task.name, task.name))
            self.tasks[sub_task.name] = sub_task

    def search_sub_tasks(self, task, ret):
        for k, v in task.sub_tasks.items():
            # Append task.
            ret.append(v)
            # Recursively add further sub-tasks.
            if len(v.sub_tasks) > 0:
                self.search_sub_tasks(v, ret)

    def act_task(self, name, states, use_exploration=True, apply_preprocessing=True,
                 time_percentage=None, propagate=True):
        """
        Passes states through the task graph. Starts acting on one task, then acts
        on all dependent sub-tasks or only on the specified tasks, depending on propagate flag.

        Args:
            name (str): Name of task to act on.
            states: Input states to root(s) of task graph.
            use_exploration (bool): Apply action exploration.
            apply_preprocessing (bool): Apply state preprocessing.
            propagate (bool): If true, propagates a task's output to all dependent sub-tasks and returns
                final output. If false, only acts on this particular tasks and no sub-tasks.

        Returns:
            dict: States for all traversed tasks (i.e. intermediate states) as nested dict..
        """
        actions = {name: self.tasks[name].act(states, use_exploration=use_exploration,
                                              apply_preprocessing=apply_preprocessing,
                                              time_percentage=time_percentage,
                                              propagate=propagate)}
        if len(actions) == 1:
            return list(actions.values())[0]
        else:
            return actions

    def observe_task(self, name, preprocessed_states, actions, internals, rewards, next_states=None, terminals=None):
        """
        Make observations for a particular task.
        """
        self.tasks[name].observe(preprocessed_states=preprocessed_states, actions=actions, internals=internals,
                                 rewards=rewards, next_states=next_states, terminals=terminals)

    def update_task(self, name, batch=None, *args):
        return self.tasks[name].update(batch, *args)

    def store_model(self, name, path):
        self.tasks[name].unwrap().store_model(path, False)

    def load_model(self, name, checkpoint_directory):
        self.tasks[name].unwrap().load_model(checkpoint_directory)

    def get_task(self, name):
        return self.tasks[name]

    def reset(self):
        for task in self.tasks.values():
            task.reset()
