from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from multiprocessing import Queue

from lift.case_studies.mongodb import mongo_system_models, mongo_model_generators, mongo_schemas
from lift.case_studies.mongodb.deprecated.combinatorial_data_loader import CombinatorialDataSource
from lift.case_studies.mongodb.deprecated.execution import executors
from lift.case_studies.mongodb.deprecated.mongo_parser import MongoParser
from lift.case_studies.mongodb.deprecated.async_controller import AsyncController


class MongoAsyncController(AsyncController):
    """
    Asynchronous MongoDB controller.
    """

    def __init__(
        self,
        agent_config,
        network_config,
        experiment_config,
        schema_config,
        host='localhost',
        result_dir='',
        duration=7200,
        executor='random',
        model_store_path='',
        model_load_path='',
        data_path='',
        store_model=False,
        load_model=False,
        serialize=False,
        serialization_path=''
    ):
        super(MongoAsyncController, self).__init__(
            agent_config=agent_config,
            network_config=network_config,
            experiment_config=experiment_config,
            schema_config=schema_config,
            host=host,
            result_dir=result_dir,
            duration=duration,
            model_store_path=model_store_path,
            model_load_path=model_load_path,
            data_path=data_path,
            store_model=store_model,
            load_model=load_model,
            serialize=serialize,
            serialization_path=serialization_path
        )
        self.logger.info("Initiating model components.")
        self.queue = Queue()
        self.schema = mongo_schemas[experiment_config['model']](
            schema_config=schema_config,
            experiment_config=experiment_config

        )
        self.log_parser = MongoParser(
            config=experiment_config,
            host=host,
            schema=self.schema,
            queue=self.queue,
            end=self.end_time
        )
        self.states_spec = self.schema.get_states_spec()
        self.actions_spec = self.schema.get_actions_spec()

        self.model_generator = mongo_model_generators[experiment_config['model']](
            schema=self.schema,
            experiment_config=experiment_config
        )
        self.system_model = mongo_system_models[experiment_config['model']](
            experiment_config=experiment_config,
            schema=self.schema,
            agent_config=agent_config,
            host=host,
            queue=self.queue,
            model_generator=self.model_generator
        )
        self.logger.info("Instantiating executor.")
        self.rl_executor = executors[executor](
            experiment_config=experiment_config,
            agent_config=agent_config,
            states_spec=self.states_spec,
            actions_spec=self.actions_spec,
            network_spec=self.network_spec,
            model=self.system_model,
            end=self.end_time,
            serialize=serialize
        )
        self.data_loader = CombinatorialDataSource(
            model_generator=self.model_generator,
            schema=self.schema
        )

    def start(self):
        self.logger.info("Starting controller demon process..")
        super(MongoAsyncController, self).start()

        # This starts the execution loop of the parser.
        self.logger.info("Starting parsing loop.")
        self.log_parser.execute()
