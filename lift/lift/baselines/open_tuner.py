import argparse
import sys
import logging
from lift.baselines.optimisation_interface import OptimisationInterface

# Avoid failing in other case studies from trying to import this Module.
try:
    import opentuner
    from opentuner.api import TuningRunManager
    from opentuner.measurement.interface import DefaultMeasurementInterface
    from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter

except ImportError:
    print("Could not import OpenTuner!")


class OpenTuner(OptimisationInterface):
    """
    Generic OpenTuner binding.
    """
    def __init__(self, experiment_config, schema, result_dir, system_model):
        self.logger = logging.getLogger(__name__)
        self.config = experiment_config
        self.schema = schema
        self.result_dir = result_dir
        self.system_environment = system_model

        # This is a trick to reset argvars - otherwise the parser below would copy all args from
        # the outer script and fail because it does not recognise them.
        # argv[0] is the program name and needs to stay in.
        sys.argv = [sys.argv[0]]
        parser = argparse.ArgumentParser(parents=opentuner.argparsers())
        args = parser.parse_args()

        # Create tasks.
        self.logger.info("Initializing OpenTuner task descriptions.")
        manipulator = self.build_tasks()
        # OpenTUNER API interface.
        interface = DefaultMeasurementInterface(args=args, manipulator=manipulator,
                                                project_name='lift', program_name='opentuner',
                                                program_version='0.1')
        self.api = TuningRunManager(interface, args)

    def build_tasks(self):
        """
        Creates a manipulator object which describes all required outputs per step.
        """
        raise NotImplementedError

    def observe(self, performance, *args, **kwargs):
        self.api.report_result(performance["desired_result"], performance["result"])

    def run(self, label, num_iterations):
        """
        Queries the OpenTuner API 'num_iteration' times
        """
        self.logger.info("Beginning OpenTuner run.")
        for i in range(num_iterations):
            self.logger.info("Initializing iteration {}".format(i))
            desired_result = self.api.get_next_desired_result()
            cfg = desired_result.configuration.data
            reward, actions, _ = self.act(cfg)

            self.logger.info("Observed episode reward: {}".format(reward))
            # Return result wrapped in OpenTuner's format.
            # Note: arg is called 'time' because runtime is minimized per default.
            # NOTE: -reward because we maximize reward, but minimize with opentuner.
            result = opentuner.resultsdb.models.Result(time=-reward)
            self.observe(performance=dict(desired_result=desired_result, result=result))

        # Now get final config.
        best = self.api.get_best_configuration()

        # Act.
        self.logger.info("Optimization completed. Evaluating final configuration.")

        # Do a final eval.
        self.eval_best(best, label)

        # Reset and teardown.
        self.system_environment.reset()
