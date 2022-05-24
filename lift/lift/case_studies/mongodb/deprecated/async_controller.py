import logging

from multiprocessing import Process

import time

from lift.case_studies.mongodb.deprecated.execution import PretrainExecutor


class AsyncController(object):

    def __init__(
        self,
        agent_config,
        network_config,
        experiment_config,
        schema_config,
        host='localhost',
        result_dir='',
        duration=3600,
        model_store_path='',
        model_load_path='',
        data_path='',
        store_model=False,
        load_model=False,
        serialize=False,
        serialization_path=''
    ):
        self.logger = logging.getLogger(__name__)
        self.agent_config = agent_config

        # General experiment/case study parameters.
        self.experiment_config = experiment_config

        # Data layout of states and actions.
        self.schema_config = schema_config

        # Network for agent.
        self.network_spec = network_config

        # Host in case of networked communication to system.
        self.host = host

        # Main result directory.
        self.result_dir = result_dir

        # Experiment duration
        self.duration = duration

        # Model path if model is imported.
        self.model_load_path = model_load_path
        self.model_store_path = model_store_path

        # Data path to pretraining data if using learning from demonstration.
        self.data_path = data_path

        # Load a model before experiment?
        self.load_model = load_model
        # Store model after experiment.
        self.store_model = store_model

        # Serialize observations for learning from trace.
        self.serialize = serialize

        # Serialization path
        self.serialization_path = serialization_path

        # Instantiate in subclass.
        self.rl_executor = None
        self.data_loader = None

        self.interval = int(experiment_config['interval'])
        self.end_time = time.time() + duration

        self.run_process = Process(target=self.run, name="controller")
        self.run_process.daemon = True

    def start(self):
        """
        Initialise query monitor, model.
        """
        # This starts the run method of this class in a separate process.
        self.run_process.start()

    def run(self):
        """
        Online control loop.
        - Initializes TensorFlow model
        - Loads pretrained model
        - Loops through intervals until specified duration is over
        - Exports results
        """
        self.logger.info("Beginning control loop.")
        self.rl_executor.init_model()

        if self.load_model:
            self.logger.info("Loading model.")
            self.rl_executor.restore_model(self.model_load_path)

            # DQFD assume that demonstration data is still available during online training
            # to sample from both online and demo data.
            if isinstance(self.rl_executor, PretrainExecutor):
                self.logger.info("Preparing serialized data.")

                data = self.data_loader.load_data(
                    data_dir=self.data_path,
                    batch_size=self.experiment_config['data_loader_batch_size'],
                    lookup_actions=False
                )
                self.rl_executor.agent.set_demonstrations(data)
                self.logger.info("Inserted serialized data.")

        while time.time() < self.end_time:
            self.rl_executor.execute()
            time.sleep(self.interval)

        kwargs = dict(
            serial_path=self.serialization_path,
            serialize=self.serialize
        )
        self.rl_executor.export(self.result_dir, kwargs)
        self.logger.info('Finished run, shutting down.')

        if self.store_model:
            self.rl_executor.store_model(self.model_store_path)
        exit(0)