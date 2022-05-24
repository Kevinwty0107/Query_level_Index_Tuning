import csv
import numpy as np

from lift.baselines.open_tuner import OpenTuner
from lift.case_studies.mongodb import mongo_model_generators
from lift.case_studies.common.baseline_util import export


# Avoid failing in other case studies from trying to import this Module.
from lift.case_studies.mongodb.query_util import check_stage
from lift.controller.system_controller import aggregation

try:
    import opentuner
    from opentuner.api import TuningRunManager
    from opentuner.measurement.interface import DefaultMeasurementInterface
    from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter

except ImportError:
    print("Could not import OpenTuner!")


class OpenTunerMongoDB(OpenTuner):
    """
    Implements an OpenTuner binding for MongoDB indexing.
    """
    def __init__(self, train_queries, test_queries, experiment_config, schema, result_dir, system_model):
        self.train_queries = train_queries
        self.test_queries = test_queries
        # Map name of query to actions.
        self.action_dict = {}
        self.action_to_table = {}

        # Map name of query to query object.
        self.query_name_dict = {}
        self.converter = mongo_model_generators[experiment_config['model']](
            schema=schema,
            experiment_config=experiment_config
        )

        # How many indices is OpenNTuner allowed to create?
        self.num_indices = experiment_config["queries_per_episode"]
        self.columns_per_index = schema.get_system_spec()['max_fields_per_index']
        self.runtime_aggregation = aggregation[experiment_config.get('runtime_aggregation', 'mean')]

        # TODO this will only work if queries are on the same table.
        # Would need to separate out per table.
        # Columns to select.
        self.collections = schema.collections
        self.indices_per_table = int(self.num_indices / len(self.collections))

        # Maps a table to the lookup dict.
        # Dict of dicts: "table_name": { 0: "column_name_1", ..} }
        self.table_column_index_to_index_tuple = {}

        # Create lookup mapping.
        for collection in self.collections:
            collection_dict = schema.collection_to_schema[collection]
            self.table_column_index_to_index_tuple[collection] = {}
            i = 0
            for field_name in collection_dict.keys():
                self.table_column_index_to_index_tuple[collection][i + 1] = (field_name, 1)
                i += 1
                self.table_column_index_to_index_tuple[collection][i + 1] = (field_name, -1)
                i += 1

        # E.g. consider a workload auf 1000 queries. Evaluating all of them could take a while.
        # We may only run a subset, e.g. 100 randomly selected queries, to evaluate our index
        # in each episode.
        self.queries_to_run_per_eval = experiment_config['open_tuner_eval_queries']

        super(OpenTunerMongoDB, self).__init__(experiment_config, schema, result_dir, system_model)
        self.logger.info("Initializing OpenTuner task descriptions.")

    def build_tasks(self):
        # Create the parameter space.
        manipulator = ConfigurationManipulator()

        # For each table:
        for collection in self.collections:
            # How many outputs per action in this table?
            num_columns_in_table = len(self.table_column_index_to_index_tuple[collection])

            # Create the allowed indices per table:
            for i in range(self.indices_per_table):
                task_name = "{}_{}".format(collection, i)
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
                self.action_to_table[task_name] = collection

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
        size = 0
        sizes = []
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
                    field = self.table_column_index_to_index_tuple[table_for_action][val][0]
                    if field not in index_fields:
                        # Look up the table, then the column corresponding to this integer.
                        system_action.append(self.table_column_index_to_index_tuple[table_for_action][val])
                        index_fields.append(field)
            if len(system_action) == 1:
                system_action[0] = (system_action[0][0], 1)
            self.system_environment.act(system_action)
            index_size, _, _ = self.system_environment.system_status()
            context.append(system_action)
            sizes.append((system_action, index_size - size))
            size = index_size
            context.append(system_action)

        # Evaluate on a random subset of the workload.
        train_query_sample = np.random.choice(self.train_queries, size=self.queries_to_run_per_eval)
        eval_runtimes = []
        for query in train_query_sample:
            runtime = self.system_environment.execute(query)
            eval_runtimes.append(runtime)
        runtime = self.runtime_aggregation(eval_runtimes)
        index_size, num_indices, final_index = self.system_environment.system_status()
        reward = self.converter.system_to_agent_reward(meta_data=dict(
            runtime=runtime, index_size=index_size))
        episode_reward += reward
        # TODO can merge context/sizes.
        return episode_reward, context, sizes

    def eval_best(self, config, label):
        # Create final configuration in system.
        ep_reward, actions, sizes = self.act(config)

        # Do a final eval of queries.
        runs = self.config['num_executions']
        runtimes = []
        for query in self.test_queries:
            runtime = []
            for _ in range(runs):
                exec_time = self.system_environment.execute(query)
                runtime.append(exec_time)
            runtimes.append((np.mean(runtime), np.std(runtime)))

        indices = actions
        num_full_indices = 0
        for query in self.test_queries:
            full_index = query.full_index_from_query()
            if full_index['index'] in indices:
                num_full_indices += 1

        # Used indices
        index_used = {}
        index_lens = {}
        index_sizes = {}
        for name, size in sizes:
            # How mongodb names indices.
            print("Index name = {}, size = {}".format(name, size))
            index_name = "_".join(["{}_{}".format(v[0], v[1]) for v in name])

            # Count usage.
            index_used[index_name] = [0, []]
            # How many fields does this index span
            index_lens[index_name] = len(name)
            index_sizes[index_name] = size

        # Explain each query -> See if there are any unused indices -> how much intersection?
        for query in self.test_queries:
            # Check index used?
            # Mark index used via counter
            # Mark unused indices.
            stats = self.system_environment.explain(query)
            self.logger.info("Analysing query = {}".format(query))

            # The layout of the response  seems to vary with MongoDB server versions.
            # Single-cursor response.
            if "cursor" in stats:
                self.logger.info("Cursor is in stats = {}, type= {}".format(stats["cursor"], type(stats["cursor"])))
                # 'cursor': 'BtreeCursor numVotes_1_titleType_-1_titleType_1
                if isinstance(stats["cursor"], str):
                    cursor_components = stats["cursor"].split(" ")
                    if cursor_components[0] == "BtreeCursor":
                        self.logger.info("BtreeCursor counted.")
                        index_name = cursor_components[1]
                        assert index_name in index_used, " Name of Btreecursor {} is not in index used, allowed are: {}".format(
                            index_name, index_used
                        )
                        # Was this index name used?
                        index_used[index_name][0] += 1
                        # Create a list of the lengths of queries using this for intersection.
                        index_used[index_name][1].append(str(len(query.query_columns)))
                    else:
                        self.logger.info("Cursor {} not counted.".format(cursor_components[0]))
            # Plan-response.
            elif "queryPlanner" in stats:
                plan_response = stats["queryPlanner"]
                assert "winningPlan" in plan_response, "Key `winningPlan` not in plan response"
                winning_plan = plan_response["winningPlan"]
                # Winning plan has a recursive structure of different stages. We are looking for
                # an index name used in a stage IXSCAN.
                if "inputStage" in winning_plan:
                    top_level_stage = winning_plan["inputStage"]
                    if top_level_stage == "SUBPLAN":
                        # Skip to subplan.
                        top_level_stage = winning_plan["inputStage"]
                    if top_level_stage["stage"] == "IXSCAN":
                        check_stage(index_used, query, top_level_stage)
                    # OR can fetch multiple indices.
                    elif top_level_stage["stage"] == "FETCH":
                        if top_level_stage["inputStage"]["stage"] == "OR":
                            stages = top_level_stage["inputStage"]["inputStages"]
                            for stage in stages:
                                if stage["stage"] == "IXSCAN":
                                    check_stage(index_used, query, stage)
                        elif top_level_stage["inputStage"]["stage"] == "IXSCAN":
                            check_stage(index_used, query, top_level_stage["inputStage"])
            else:
                self.logger.info("No cursor found, dumping raw stats: {} ".format(stats))

        # Export index usage.
        with open(self.result_dir + '/' + label + '_index_usage.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for name, value in index_used.items():
                # index len # name of index # times used # Size of queries using # num full indices <- repeated.
                # 1_2_3_1..
                if value[1]:
                    sizes_string = "_".join(value[1])
                else:
                    sizes_string = "0"
                writer.writerow([index_lens[name], name, value[0], sizes_string, num_full_indices, index_sizes[name]])

        # Export results.
        index_size, num_indices, final_index = self.system_environment.system_status()
        self.logger.info("Exporting final results.")
        export(self.result_dir, label, runtimes, index_size, num_indices, final_index, self.train_queries, actions)
