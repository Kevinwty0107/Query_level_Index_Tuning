from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from lift.backends import set_learning_rate
from lift.case_studies.common.index_net import build_index_net
from lift.case_studies.mongodb.imdb_synthetic_workload import IMDBSyntheticWorkload
from lift.case_studies.mongodb.mongodb_data_source import MongoDBDataSource
from lift.case_studies.mongodb import mongo_model_generators, FieldPositionSchema, mongo_demo_rules
from lift.case_studies.mongodb.fixed_imdb_workload import FixedIMDBWorkload
from lift.pretraining.pretrain_controller import PretrainController
from lift.rl_model.task import Task


class MongoPretrainController(PretrainController):

    def __init__(
        self,
        agent_config,
        network_config,
        experiment_config,
        schema_config,
        use_full=True,
        result_dir=None,
        model_path=None,
        load_model=False,
        training_dir=None,
        test_dir=None,
        blackbox_mode=False,
        fixed_workload=False
    ):
        """
        Creates a controller for MongoDB pre-training.

        Args:
            agent_config (dict): Agent config dict.
            network_config (dict): Network config dict.
            experiment_config (dict): Experiment config dict.
            schema_config (dict): Schema config dict.
            result_dir (str): Path to directory to store results.
            model_path (str): Path to save model to.
            load_model (bool): Whether to load a model from an existing model_path.
            training_dir (str): Directory containing training trace.
            test_dir (str): Optional Directory containing test trace.
        """
        super(MongoPretrainController, self).__init__(
            agent_config=agent_config,
            experiment_config=experiment_config,
            model_path=model_path,
            load_model=load_model,
            result_dir=result_dir,
            blackbox_mode=blackbox_mode
        )

        self.training_dir = training_dir
        self.test_dir = test_dir
        self.experiment_config = experiment_config
        self.schema_config = schema_config

        schema = FieldPositionSchema(
            schema_config=schema_config,
            schema_spec=experiment_config["schema_spec"],
            mode=self.state_mode
        )
        self.states_spec = schema.states_spec
        self.actions_spec = schema.actions_spec

        # Specifying network here because I need the vocab size.
        vocab_size = len(schema.get_system_spec()['input_vocabulary'].keys())
        layer_size = experiment_config['layer_size']
        self.logger.info("Initialising embedding with vocabsize = {}".format(vocab_size))
        self.logger.info("Vocab = {}".format(schema.get_system_spec()['input_vocabulary']))
        agent_config["network_spec"] = build_index_net(
                state_mode=self.state_mode,
                states_spec=self.states_spec,
                embed_dim=experiment_config["embedding_size"],
                vocab_size=vocab_size,
                layer_size=layer_size)

        self.converter = mongo_model_generators[experiment_config['model']](
            schema=schema,
            experiment_config=experiment_config
        )

        # Use separate learning rate
        if 'pretrain_learning_rate' in self.experiment_config:
            self.logger.info('Setting pretrain learning rate to {}'.format(
                self.experiment_config['pretrain_learning_rate']))
            set_learning_rate(self.agent_config, self.experiment_config['pretrain_learning_rate'])

        # Create RLgraph agent from spec.
        self.agent_type = self.agent_config["type"]

        task = Task(self.agent_config, self.states_spec, self.actions_spec)
        self.task_graph.add_task(task)

        self.data_source = MongoDBDataSource(
            converter=self.converter,
            schema=schema
        )

        workload_spec = {
            "num_selections": schema.max_fields_per_index
        }
        if fixed_workload is True:
            self.workload_gen = FixedIMDBWorkload()
        else:
            self.workload_gen = IMDBSyntheticWorkload(workload_spec)

        self.demo_rules = []
        for rule, spec in experiment_config["demo_rules"].items():
            self.demo_rules.append(mongo_demo_rules[rule](reward=spec["reward"], margin=spec["margin"]))
