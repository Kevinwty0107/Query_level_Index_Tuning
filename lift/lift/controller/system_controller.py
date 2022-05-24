import logging
import numpy as np

aggregation = {
    "mean": np.mean,
    "percentile_99": lambda x: np.percentile(x, 99),
    "percentile_90": lambda x: np.percentile(x, 90)
}


class SystemController(object):

    def __init__(
        self,
        agent_config,
        experiment_config,
        network_config=None,
        result_dir='',
        model_store_path='',
        model_load_path='',
        store_model=False,
        load_model=False
    ):
        """
        Creates a system controller.

        Args:
            agent_config (dict): RLgraph agent configuration
            network_config (list): Optional List of neural network layer descriptors. If none, define
                in controller based on other input variables.
            experiment_config (dict): General experiment settings.
            result_dir (str): Path to directory for result files.
            model_store_path (str): Path to export directory.
            model_load_path (str): Path to TF model checkpoint.
            store_model (bool): If true, export model to `model_store_path` after training.
            load_model (bool): If true, import model from `model_load_path` before training.
        """
        self.logger = logging.getLogger(__name__)
        self.state_mode = experiment_config.get('state_mode', 'default')
        self.agent_config = agent_config
        if "explore_timesteps" in experiment_config:
            self.max_steps = experiment_config["explore_timesteps"]
        else:
            self.max_steps = None
        self.training_reward = experiment_config.get('training_reward', 'incremental')
        self.runtime_aggregation = aggregation[experiment_config.get('runtime_aggregation', 'mean')]
        self.runtime_cache = {}
        self.updating = None
        self.task_graph = None
        self.system_environment = None

        self.steps_before_update = None
        self.update_interval = None
        self.update_steps = None
        self.sync_interval = None
        self.episodes_since_update = 0
        self.time_step = 0.0

        # General experiment/case study parameters.
        self.experiment_config = experiment_config

        # Network for agent.
        self.network_spec = network_config

        # IO parameters.
        self.model_store_path = model_store_path
        self.model_load_path = model_load_path

        self.store_model = store_model
        self.load_model = load_model

        self.result_dir = result_dir

    def run(self, *args, **kwargs):
        """
        Runs an entire experiment lifecycle which may include training, testing
        and evaluation of different baselines.
        """
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """
        Executes online training by interacting with an environment through a `SystemEnvironment`.

        Args:
            *args: Training args.
            **kwargs: Training kwargs.
        """
        raise NotImplementedError

    def evaluate_tf_model(self, path, *args, **kwargs):
        """
        For a given trained TensorFlow model, import model and data, and run an evaluation
        on the model without further training.

        Args:
            path (str): Path to TensorFlow checkpoint.
        """
        pass

    def init_workload(self, *args, **kwargs):
        """
        Optionally prepares workload, e.g. by importing queries or a generating a task schedule to
        simulate.
        """
        pass

    def update_if_necessary(self):
        """
        Calls update on the agent according to the update schedule set for this worker.

        #Args:
        #    timesteps_executed (int): Timesteps executed thus far.

        Returns:
            float: The summed up loss (over all self.update_steps).
        """
        if self.updating:
            # Are we allowed to update?
            agent = self.task_graph.get_task("").unwrap()
            if agent.timesteps > self.steps_before_update and \
                    (agent.observe_spec["buffer_enabled"] is False or  # No update before some data in buffer
                     agent.timesteps >= agent.observe_spec["buffer_size"]):
                # Updating according to one update mode:
                if self.update_mode == "time_steps" and agent.timesteps % self.update_interval == 0:
                    loss = self.execute_update()
                    self.logger.info("Finished time-step based update, loss = {}".format(loss))
                    return loss
                elif self.update_mode == "episodes" and self.episodes_since_update == self.update_interval:
                    # Do not do modulo here - this would be called every step in one episode otherwise.
                    loss = self.execute_update()
                    self.episodes_since_update = 0
                    self.logger.info("Finished episode-based update, loss = {}".format(loss))
                    return loss
        return None

    def execute_update(self):
        loss = 0
        for _ in range(self.update_steps):
            ret = self.task_graph.update_task(name="")
            if isinstance(ret, tuple):
                loss += ret[0]
            else:
                loss += ret
        return loss

    def set_update_schedule(self, update_schedule=None):
        """
        Sets the controllers's update schedule.

        Args:
            update_schedule (Optional[dict]): Update parameters. If None, the worker only performs rollouts.
                Expects keys 'update_interval' to indicate how frequent update is called, 'num_updates'
                to indicate how many updates to perform every update interval, and 'steps_before_update' to indicate
                how many steps to perform before beginning to update.
        """
        if update_schedule is not None:
            self.updating = True
            self.steps_before_update = update_schedule['steps_before_update']
            self.update_interval = update_schedule['update_interval']
            self.update_steps = update_schedule['update_steps']
            self.sync_interval = update_schedule['sync_interval']

            # Interpret update interval as n time-steps or n episodes.
            self.update_mode = update_schedule.get("update_mode", "time_steps")

    def reset_system(self):
        """
        Resets system after acting.

        Important: this only resets indices that were created in this program, not any prior existing indices.
        """
        self.logger.info("Resetting system.")
        self.system_environment.reset()

    def import_model(self, checkpoint_dir):
        """
        Imports a model from a checkpoint directory path.
        """
        self.task_graph.load_model(name="", checkpoint_directory=checkpoint_dir)
