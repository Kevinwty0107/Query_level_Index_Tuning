from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from lift.backends import set_learning_rate
from lift.case_studies.common.index_net import build_index_net
from rlgraph.agents import Agent

from lift.case_studies.mysql import mysql_demo_rules
from lift.case_studies.mysql.mysql_converter import MySQLConverter
from lift.case_studies.mysql.mysql_data_source import MySQLDataSource
from lift.case_studies.mysql.mysql_schema import MySQLSchema
from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload
from lift.case_studies.mysql.tpch_util import SCALE_FACTOR
from lift.pretraining.pretrain_controller import PretrainController
from lift.pretraining.simple_evaluator import SimpleEvaluator
from lift.rl_model.task import Task


class MySQLPretrainController(PretrainController):

    def __init__(
        self,
        agent_config,
        network_config,
        experiment_config,
        schema_config,
        result_dir=None,
        model_path=None,
        load_model=False,
        training_dir=None,
        test_dir=None,
        blackbox_mode=False
     ):
        """
        Creates a controller for MySQL pre-training.

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
        super(MySQLPretrainController, self).__init__(
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

        schema = MySQLSchema(
            schema_config=schema_config,
            mode=self.state_mode
        )
        self.states_spec = schema.states_spec
        self.actions_spec = schema.actions_spec

        # Specifying network here because I need the vocab size.
        vocab_size = len(schema.get_system_spec()['input_vocabulary'].keys())
        layer_size = experiment_config['layer_size']
        agent_config["network_spec"] = build_index_net(
                state_mode=self.state_mode,
                states_spec=self.states_spec,
                embed_dim=experiment_config["embedding_size"],
                vocab_size=vocab_size,
                layer_size=layer_size)

        self.converter = MySQLConverter(
            schema=schema,
            experiment_config=experiment_config
        )

        tables = schema_config.get("tables", None)
        num_selections = schema_config.get("num_selections", 3)

        if tables is None:
            tables = ["ORDERS", "LINEITEM", "PART"]

        workload_spec = {
            "tables": tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": num_selections
        }
        self.workload_gen = TPCHSyntheticWorkload(workload_spec=workload_spec)

        # Note that the data loader can be stateful, thus separate for train and test.
        self.data_source = MySQLDataSource(
            converter=self.converter,
            schema=schema
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

        self.demo_rules = []
        for rule, spec in experiment_config["demo_rules"].items():
            self.demo_rules.append(mysql_demo_rules[rule](reward=spec["reward"], margin=spec["margin"]))
