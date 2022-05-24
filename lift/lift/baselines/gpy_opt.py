from lift.baselines.optimisation_interface import OptimisationInterface
import logging

try:
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("Could not import GPyOpt!")


class GPyOpt(OptimisationInterface):
    """
    Interfaces the Bayesian optimisation library GPyOpt:

    https://github.com/SheffieldML/GPyOpt

    API instructions, examples at:

    http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_reference_manual.ipynb
    """
    def __init__(self, experiment_config, schema, result_dir, system_model):
        self.logger = logging.getLogger(__name__)
        self.config = experiment_config
        self.schema = schema
        self.result_dir = result_dir
        self.system_model = system_model
        self.opt = self.build_tasks()

    def build_tasks(self):
        """
        Builds an optimizer object.
        """
        raise NotImplementedError

    def run(self, label, num_iterations):
        """
        Queries the OpenTuner API 'num_iteration' times
        """
        self.logger.info("Beginning GPyOpt run.")
        self.opt.run_optimization(max_time=num_iterations)

        # Now get best guess.
        # TODO what format is this?
        best = self.opt.x_opt

        # Act.
        self.logger.info("Optimization completed. Evaluating final configuration.")

        # Do a final eval.
        self.eval_best(best, label)

        # Reset and teardown.
        self.system_model.reset()
