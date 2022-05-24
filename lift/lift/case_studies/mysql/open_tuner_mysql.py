import numpy as np

from lift.baselines.open_tuner import OpenTuner
from lift.case_studies.mysql.mysql_converter import MySQLConverter
from lift.case_studies.common.baseline_util import export


# Avoid failing in other case studies from trying to import this Module.
from lift.case_studies.mysql.tpch_util import tpch_table_columns
from lift.controller.system_controller import aggregation

try:
    import opentuner
    from opentuner.api import TuningRunManager
    from opentuner.measurement.interface import DefaultMeasurementInterface
    from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter

except ImportError:
    print("Could not import OpenTuner!")


class OpenTunerMySQL(OpenTuner):
    """
    Implements an OpenTuner binding for MySQL indexing.
    """
    def __init__(self, train_queries, test_queries, experiment_config, schema, result_dir, system_model):
        self.train_queries = train_queries
        self.test_queries = test_queries

        # Map name of task to actions.
        self.action_dict = {}
        self.action_to_table = {}

        # How many indices is OpenNTuner allowed to create?
        self.num_indices = experiment_config["queries_per_episode"]
        self.columns_per_index = schema.get_system_spec()['max_fields_per_index']
        self.runtime_aggregation = aggregation[experiment_config.get('runtime_aggregation', 'mean')]

        # TODO this will only work if queries are on the same table.
        # Would need to separate out per table.
        # Columns to select.
        self.tables = schema.tables
        self.indices_per_table = int(self.num_indices / len(self.tables))

        # Maps a table to the lookup dict.
        # Dict of dicts: "table_name": { 0: "column_name_1", ..} }
        self.table_column_index_to_tuple = {}

        # Create lookup mapping.
        for table in self.tables:
            table_dict = tpch_table_columns[table]
            self.table_column_index_to_tuple[table] = {}
            i = 0
            for column_name in table_dict.keys():
                self.table_column_index_to_tuple[table][i + 1] = (column_name, "ASC")
                i += 1
                self.table_column_index_to_tuple[table][i + 1] = (column_name, "DESC")
                i += 1

        # E.g. consider a workload auf 1000 queries. Evaluating all of them could take a while.
        # We may only run a subset, e.g. 100 randomly selected queries, to evaluate our index
        # in each episode.
        self.queries_to_run_per_eval = experiment_config['open_tuner_eval_queries']

        self.converter = MySQLConverter(
            schema=schema,
            experiment_config=experiment_config
        )
        super(OpenTunerMySQL, self).__init__(experiment_config, schema, result_dir, system_model)

        self.logger.info("Initializing OpenTuner task descriptions.")

    def build_tasks(self):
        # Create the parameter space.
        manipulator = ConfigurationManipulator()

        # For each table:
        for table in self.tables:
            # How many outputs per action in this table?
            num_columns_in_table = len(self.table_column_index_to_tuple[table])

            # Create the allowed indices per table:
            for i in range(self.indices_per_table):
                task_name = "{}_{}".format(table, i)
                # Define 3 actions, save names.
                action_list = []

                # We need to store the 3 together to form compound indices.
                for k in range(self.columns_per_index):
                    action_name = "index_field{}".format(k)

                    # Create unique name for each output: table_index_i_index_field_k
                    prefixed_name = "{}+{}".format(task_name, action_name)

                    # Min and max are inclusive.
                    action_list.append(prefixed_name)
                    # Prefix with query or we try to create the same parameter many times.
                    manipulator.add_parameter(IntegerParameter(prefixed_name, 0, num_columns_in_table))

                # Now map: table_index_i -> its k columns so they can be grouped into a single index.
                self.action_dict[task_name] = action_list
                # Save table for this compound index.
                self.action_to_table[task_name] = table

        self.logger.info("Initialized actions: {}".format(self.action_dict))
        return manipulator

    def observe(self, performance, *args, **kwargs):
        self.api.report_result(performance["desired_result"], performance["result"])

    def act(self, states, *args, **kwargs):
        cfg = states
        context = []
        self.system_environment.reset()

        episode_reward = 0
        # Iterate over defined actions.
        for task_name, index_actions in self.action_dict.items():

            # Each action is a list of the parameter names for one compound index.
            action_values = []
            for action_name in index_actions:
                action_value = cfg[action_name]
                action_values.append(action_value)

            # Look up columns in the relevant table:
            table_for_action = self.action_to_table[task_name]
            system_action = []
            index_fields = []
            for val in action_values:
                # 0 == no op.
                if val != 0:
                    field = self.table_column_index_to_tuple[table_for_action][val][0]
                    if field not in index_fields:
                        # Look up the table, then the column corresponding to this integer.
                        system_action.append(self.table_column_index_to_tuple[table_for_action][val])
                        index_fields.append(field)
            if len(system_action) == 1:
                system_action[0] = (system_action[0][0], "ASC")
            self.system_environment.act(dict(index=system_action, table=table_for_action))
            context.append(system_action)

        # Evaluate on a random subset of the workload.
        train_query_sample = np.random.choice(self.train_queries, size=self.queries_to_run_per_eval)
        eval_runtimes = []
        for query in train_query_sample:
            query_string, query_args = query.sample_query()
            runtime = self.system_environment.execute(query_string, query_args)
            eval_runtimes.append(runtime)

        index_size, num_indices, final_index = self.system_environment.system_status()
        runtime = self.runtime_aggregation(eval_runtimes)
        reward = self.converter.system_to_agent_reward(meta_data=dict(
            runtime=runtime, index_size=index_size))
        episode_reward += reward
        return episode_reward, context, []

    def eval_best(self, config, label):
        # Create final configuration in system.
        ep_reward, actions, _ = self.act(config)

        # Do a final eval of queries.
        runs = self.config['num_executions']
        runtimes = []
        for query in self.test_queries:
            runtime = []
            for _ in range(runs):
                query_string, query_args = query.sample_query()
                exec_time = self.system_environment.execute(query_string, query_args)
                runtime.append(exec_time)
            runtimes.append((np.mean(runtime), np.std(runtime)))

        # Export results.
        index_size, num_indices, final_index = self.system_environment.system_status()
        self.logger.info("Exporting final results.")
        export(self.result_dir, label, runtimes, index_size, num_indices, final_index, self.train_queries, actions)
