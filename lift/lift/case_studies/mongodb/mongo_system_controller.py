import sys
from copy import deepcopy

from lift.case_studies.common.index_net import build_index_net
from lift.case_studies.mongodb.imdb_synthetic_workload import IMDBSyntheticWorkload
from lift.case_studies.mongodb.mongodb_data_source import MongoDBDataSource

from lift.case_studies.mongodb import FieldPositionSchema, FieldPositionConverter
from lift.case_studies.mongodb.fixed_imdb_workload import FixedIMDBWorkload
from lift.case_studies.mongodb.mongodb_system_environment import MongoDBSystemEnvironment
from lift.case_studies.mongodb.query_util import check_stage
from lift.controller.system_controller import SystemController
import time
import numpy as np
import csv

from lift.rl_model.task import Task
from lift.rl_model.task_graph import TaskGraph
from lift.util.parsing_util import sort_input


class MongoSystemController(SystemController):
    """
    Mongo online controller - executes a set of training or test queries for
    a number of specified steps on a MySQL instance and records the result.
    """

    def __init__(
            self,
            agent_config,
            network_config,
            experiment_config,
            schema_config,
            host="localhost",
            result_dir="",
            demo_data_label="",
            model_store_path="",
            model_load_path="",
            blackbox_optimization=False,
            fixed_workload=False
    ):
        super(MongoSystemController, self).__init__(
            agent_config=agent_config,
            experiment_config=experiment_config,
            network_config=network_config,
            result_dir=result_dir,
            model_store_path=model_store_path,
            model_load_path=model_load_path,
        )
        self.schema_config = schema_config
        self.demo_data_label = demo_data_label
        self.schema = FieldPositionSchema(
            schema_config=schema_config,
            schema_spec=experiment_config["schema_spec"],
            mode=self.state_mode
        )
        self.blackbox_optimization = blackbox_optimization

        self.states_spec = self.schema.states_spec
        self.actions_spec = self.schema.actions_spec
        vocab_size = len(self.schema.get_system_spec()['input_vocabulary'].keys())
        layer_size = experiment_config['layer_size']
        self.logger.info("Initialising embedding with vocabsize = {}".format(vocab_size))
        self.logger.info("Vocab = {}".format(self.schema.get_system_spec()['input_vocabulary']))

        agent_config["network_spec"] = build_index_net(
                state_mode=self.state_mode,
                states_spec=self.states_spec,
                embed_dim=experiment_config["embedding_size"],
                vocab_size=vocab_size,
                layer_size=layer_size)

        # Converters states and actions.
        self.converter = FieldPositionConverter(
            schema=self.schema,
            experiment_config=experiment_config
        )

        # Workload.
        # Only used for black-box search.
        self.max_reward = -1000000
        self.best_train_index_set = []

        # Keep track of the size of individual indices.
        self.last_size = 0
        self.name_to_sizes = {}

        # Workload spec.
        self.num_executions = experiment_config['num_executions']
        self.training_episodes = experiment_config["training_episodes"]
        self.queries_per_episode = experiment_config["queries_per_episode"]

        # On how many different episodes do we evaluate the trained model?
        self.test_episodes = experiment_config["test_episodes"]
        workload_spec = {
            "num_selections": self.schema.max_fields_per_index
        }
        if fixed_workload is True:
            self.workload = FixedIMDBWorkload
        else:
            self.workload = IMDBSyntheticWorkload(workload_spec)

        # Executes changes on system.
        self.system_environment = MongoDBSystemEnvironment(
            experiment_config=experiment_config,
            host=host
        )

        # Loads trace data.
        self.data_source = MongoDBDataSource(
            converter=self.converter,
            schema=self.schema
        )

        # Creates RLgraph agent from spec.
        self.task_graph = TaskGraph()
        task = Task(
            self.agent_config,
            state_space=self.states_spec,
            action_space=self.actions_spec
        )
        self.task_graph.add_task(task)
        self.set_update_schedule(self.agent_config["update_spec"])
        # Test-time similarity.
        self.query_hashes = set()

    def import_queries(self, label=""):
        return self.data_source.load_data(data_dir=self.result_dir, label=label)

    def generate_workload(self, export=True, label="mysql"):
        """
        Generates a set of training and test queries.

        Args:
            export (bool): If true, serialise generated queries to 'self.result_dir'.
            label (str): Workload label.

        Returns:
            Tuple: Training queries, test queries.
        """
        if isinstance(self.workload, IMDBSyntheticWorkload):
            if self.blackbox_optimization:
                num_train_queries = self.queries_per_episode
                train_queries = [self.workload.generate_query_template() for _ in range(num_train_queries)]
                test_queries = [deepcopy(q) for q in train_queries]
            else:
                num_train_queries = self.training_episodes * self.queries_per_episode
                train_queries = [self.workload.generate_query_template() for _ in range(num_train_queries)]

                num_test_queries = self.test_episodes * self.queries_per_episode
                test_queries = [self.workload.generate_query_template() for _ in range(num_test_queries)]

            if export:
                self.data_source.export_data(data=train_queries, data_dir=self.result_dir, label=label + "_train")
                self.data_source.export_data(data=test_queries, data_dir=self.result_dir, label=label + "_test")
        else:
            # Otherwise assume fixed workload.
            train_queries = self.workload.define_train_queries(self.queries_per_episode)
            test_queries = self.workload.define_test_queries(self.queries_per_episode)
        return train_queries, test_queries

    def train(self, queries, label):
        """
        Executes training and exports results.

        Args:
            train_queries (list): List of train queries. Note that these are all queries across
                all episodes, so each episode will just consist of one slice.
            label (str): Training label.
        """
        all_rewards = []
        episode_rewards = []
        episode_times = []
        # Time spent waiting on action execution.
        system_waiting_times = []
        # Time spent interacting with agent via parsing, act() and observe().
        agent_interaction_times = []
        evaluation_times = []
        train_losses = []
        self.runtime_cache = {}
        # TODO this should be done in rlgraph reset -> atm only preprocessor reset.
        self.task_graph.get_task("").unwrap().timesteps = 0
        if not self.max_steps:
            self.max_steps = len(queries)
        self.time_step = 0.0

        for current_episode in range(self.training_episodes):
            self.task_graph.reset()
            self.logger.info("Starting training episode {}.".format(current_episode))
            episode_start = time.monotonic()
            self.system_environment.reset()
            context = []
            query_index = 0
            episode_reward = 0
            episode_system_waiting_time = 0
            episode_agent_interaction_time = 0
            runtimes = []
            total_size = 0

            # Select next set of queries.
            if self.blackbox_optimization:
                episode_queries = queries
            else:
                episode_queries = queries[current_episode * self.queries_per_episode:
                                          (current_episode + 1) * self.queries_per_episode]
            episode_queries = sort_input(episode_queries)
            last_index = len(episode_queries) - 1

            for query in episode_queries:
                self.logger.info("Evaluating query = {}".format(query_index))
                interaction_start = time.monotonic()
                state = self.converter.system_to_agent_state(
                    query=query, system_context=dict(index_columns=context)
                )
                self.query_hashes.add(str(query.as_tokens()))
                action = self.task_graph.act_task("", states=state.get_value(), apply_preprocessing=True,
                                                  use_exploration=True, time_percentage=self.time_step / self.max_steps)
                system_action = self.converter.agent_to_system_action(
                    actions=action,
                    meta_data=dict(query_columns=query.query_columns)
                )
                acting_time = time.monotonic() - interaction_start
                system_runtime = self.system_environment.act(system_action)
                episode_system_waiting_time += system_runtime

                # TODO unify between mongo/mysql.
                noop_action = self.system_environment.is_noop(system_action)
                runtime = self.get_training_runtimes(episode_queries, query, noop_action)
                index_size, num_indices, final_index = self.system_environment.system_status()

                update_start = time.monotonic()
                reward = self.converter.system_to_agent_reward(meta_data=dict(
                    runtime=runtime, index_size=index_size))
                # Total reward + its elementns.
                all_rewards.append((reward, runtime, index_size))
                query_index += 1
                terminal = query_index == last_index
                context.append(system_action)
                # Last entry: next state = state
                next_state_index = query_index if query_index <= last_index else query_index - 1

                next_query = episode_queries[next_state_index]
                next_state = self.converter.system_to_agent_state(
                    query=next_query, system_context=dict(index_columns=context)
                )
                self.task_graph.observe_task(name="", preprocessed_states=state.get_value(),
                                             actions=action, internals=[], rewards=reward,
                                             next_states=next_state.get_value(),
                                             terminals=terminal)
                loss = self.update_if_necessary()
                if loss is not None:
                    train_losses.append(loss)

                update_time = time.monotonic() - update_start
                episode_agent_interaction_time += (acting_time + update_time)

                episode_reward += reward
                runtimes.append(runtime)
                total_size = index_size
                self.time_step += 1.0
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdin.flush()

            #  N.b. we are evaluating on separately sampled test set ->
            # Not direct search but generalisation.
            if episode_reward > self.max_reward:
                self.max_reward = episode_reward
                self.best_train_index_set = context

            episode_times.append(time.monotonic() - episode_start)
            self.logger.info("Finished episode: episode reward: {}, mean runtime: {}, total size: {}.".format(
                episode_reward, np.mean(runtimes), total_size))
            sys.stdin.flush()
            episode_rewards.append(episode_reward)
            system_waiting_times.append(episode_system_waiting_time)
            agent_interaction_times.append(episode_agent_interaction_time)
            evaluation_times.append(np.sum(runtimes))

        # Export episode rewards and timings.
        np.savetxt(self.result_dir + '/' + label + 'episode_rewards.txt',
                   np.asarray(episode_rewards), delimiter=',')
        np.savetxt(self.result_dir + '/' + label + 'step_rewards.txt',
                   np.asarray(all_rewards), delimiter=',')
        np.savetxt(self.result_dir + '/' + label + 'train_losses.txt',
                   np.asarray(train_losses), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + label + 'episode_durations.txt',
                   np.asarray(episode_times), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + label + 'system_waiting_times.txt',
                   np.asarray(system_waiting_times), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + label + 'agent_action_times.txt'
                   , np.asarray(agent_interaction_times), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + label + 'evaluation_times.txt',
                   np.asarray(evaluation_times), delimiter=',')

    def get_training_runtimes(self, episode_queries, query, noop):
        if self.training_reward == 'full':
            query_runtimes = []
            for ep_query in episode_queries:
                query_runtimes.append(self.system_environment.execute(ep_query))
            runtime = self.runtime_aggregation(query_runtimes)
        elif self.training_reward == 'differential':
            query_runtimes = []
            for i, ep_query in enumerate(episode_queries):
                if str(ep_query.query_dict) in self.runtime_cache:
                    # Common columns -> re-execute. Unless action was noop -> do not reexecute.
                    if not noop and set(ep_query.query_columns).intersection(set(query.query_columns)):
                        # self.logger.info('Runtime cached but intersecting columnss ep query = {}, query = {}'.
                        #                  format(ep_query.query_columns, query.query_columns))
                        t = self.system_environment.execute(ep_query)
                        # Cache runtime of that query.
                        self.runtime_cache[str(ep_query.query_dict)] = t
                        query_runtimes.append(t)
                    else:
                        # self.logger.info('Using cached runtime for episode query = {}'.format(i))
                        # No common columns -> use  cached value.
                        query_runtimes.append(self.runtime_cache[str(ep_query.query_dict)])
                else:
                    # self.logger.info('No cache entry for episode query = {}'.format(i))
                    t = self.system_environment.execute(ep_query)
                    # Cache runtime of that query.
                    self.runtime_cache[str(ep_query.query_dict)] = t
                    query_runtimes.append(t)
            runtime = self.runtime_aggregation(query_runtimes)
        elif self.training_reward == 'differential_uncached':
            query_runtimes = []
            for ep_query in episode_queries:
                query_runtimes.append(self.system_environment.execute(ep_query))
            runtime = self.runtime_aggregation(query_runtimes)
        else:
            runtime = self.system_environment.execute(query)
        return runtime

    def act(self, test_queries, demo_rule=None, index_set=None):
        """
        Acts on each query by stepping through the episode (without further training.

        Args:
            test_queries (list): List of test queries.
            demo_rule (DemonstrationRule): Optional rule to use for index creation.
            index_set (list): Optional list of indices to use.

        Returns:
            list: Actions performed.
        """
        assert demo_rule is None or index_set is None, "Can only use one of demo rule or index set (or neither)" \
                                                       "to generate indices."
        # Sort by len
        test_queries = sort_input(test_queries)

        context = []
        system_actions = []
        size = 0
        if demo_rule is not None:
            self.logger.info("Evaluating with demo rule.")
            # Create all indices according to demo rule.
            for query in test_queries:
                system_action = demo_rule.generate_demonstration(states=query, context=context)
                self.system_environment.act(system_action)
                index_size, _, _ = self.system_environment.system_status()
                context.append(system_action["index"])
                system_actions.append((system_action["index"], index_size - size))
                size = index_size
        elif index_set is not None:
            self.logger.info("Acting with provided index set {}.".format(index_set))

            for index_action in index_set:
                self.system_environment.act(index_action)
                index_size, _, _ = self.system_environment.system_status()
                # Store deltas.
                system_actions.append((index_action, index_size - size))
                size = index_size
        else:
            self.logger.info("Acting with current model.")

            for query in test_queries:
                state = self.converter.system_to_agent_state(
                    query=query, system_context=dict(index_columns=context)
                )
                action = self.task_graph.act_task("", states=state.get_value(), apply_preprocessing=True,
                                                  use_exploration=False)
                system_action = self.converter.agent_to_system_action(
                    actions=action,
                    meta_data=dict(query_columns=query.query_columns)
                )
                self.system_environment.act(system_action)
                index_size, _, _ = self.system_environment.system_status()
                # Store deltas.
                system_actions.append((system_action, index_size - size))
                context.append(system_action)
                size = index_size
        return system_actions

    def restore_workload(self, train_label, test_label):
        """
        Restores a workload from serialised version.

        """
        train_queries = self.data_source.load_data(self.result_dir, label=train_label)
        test_queries = self.data_source.load_data(self.result_dir, label=test_label)

        return train_queries, test_queries

    def evaluate(self, queries, label, actions=None, export_evaluation=True, test_similarity=False):
        """
        Evaluates a set of queries. This assumes indices have already been created by whatever means
        (rule, pretrained model, online trained model, etc.).

        Args:
            queries (list): List of queries to execute.
            label (str): Label for this evaluation. Results will be saved under this label.
            actions (list): Optional actions used for this evaluation, and their sizes.
            export_evaluation (bool):
            test_similarity: If  True, test  hashes against training set.
        """
        runtimes = []
        index_performance = {"queries": {}}
        for query in queries:
            runtime = []
            for _ in range(self.num_executions):
                exec_time = self.system_environment.execute(query)
                runtime.append(exec_time)
            runtimes.append((np.mean(runtime), np.std(runtime)))
            index_performance["queries"][query] = runtime

        index_size, num_indices, final_index = self.system_environment.system_status()
        index_performance["actions"] = actions
        index_performance["status"] = [index_size, num_indices, final_index]
        if test_similarity:
            tokens_seen = 0
            # Was this exact token combination seen during training?
            for query in queries:
                if str(query.as_tokens()) in self.query_hashes:
                    tokens_seen += 1

            similarity = float(tokens_seen) / len(queries)
            np.savetxt(self.result_dir + '/' + label + 'test_similarity.txt',
                       np.asarray([similarity]), delimiter=',')
        if export_evaluation:
            self.export(label=label, runtimes=runtimes, index_size=index_size, num_indices=num_indices,
                        final_index=final_index, queries=queries, indices=actions)
        return index_performance

    def export(self, label, runtimes, index_size, num_indices, final_index, queries=None, indices=None):
        path = self.result_dir + '/' + label

        # Export runtime tuples.
        with open(path + '_runtimes.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for row in runtimes:
                writer.writerow(row)

        # Export queries with their respective indices and runtimes for easier manual analysis.
        with open(path + '_query_index_runtime.csv', 'a', newline='') as f:
            writer = csv.writer(f, delimiter='-')
            for i in range(len(runtimes)):
                runtime_tuple = runtimes[i]
                query = queries[i]
                if indices:
                    index = indices[i]
                else:
                    index = 'None'
                writer.writerow([runtime_tuple[0], runtime_tuple[1], query.as_csv_row(), index])

        # Export index size.
        with open(path + '_index_size.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([index_size])
            writer.writerow([num_indices])

        final_index_path = self.result_dir + '/final_index/' + label + '_final_index.json'
        with open(final_index_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for index_key, value in final_index.items():
                writer.writerow([str(index_key), str(value)])

    def analyse_evaluation(self, index_performance, label=""):
        """
        Analyses indexing decisions w.r.t performance.

        1. How many full indices were created?
        2. How much intersection was leveraged? -> explain?
        3. Was there an unused index?

        Args:
            index_performance (dict): Contains queries, actions (indices), an runtimes + system status.
        """
        # Full indices -> check query columns against index columns for exact matches?
        queries = index_performance["queries"]
        indices = index_performance["actions"]
        num_full_indices = 0
        for query in queries:
            full_index = query.full_index_from_query()
            if full_index['index'] in indices:
                num_full_indices += 1

        # Used indices
        index_used = {}
        index_lens = {}
        index_sizes = {}
        self.logger.info("Beginning index analysis, retrieved indices:")
        self.logger.info(indices)
        for name, size in indices:
            # How mongodb names indices.
            print("Index name = {}, size = {}".format(name, size))
            index_name = "_".join(["{}_{}".format(v[0], v[1]) for v in name])

            # Count usage.
            index_used[index_name] = [0, []]
            # How many fields does this index span
            index_lens[index_name] = len(name)
            index_sizes[index_name] = size

        # Explain each query -> See if there are any unused indices -> how much intersection?
        all_stats = []
        for query in queries:
            # Check index used?
            # Mark index used via counter
            # Mark unused indices.
            stats = self.system_environment.explain(query)
            # self.logger.info("Analysing query = {}".format(query))

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
                all_stats.append(str(stats))
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

        # Dump raw stats.
        with open(self.result_dir + '/' + label + '_raw_explain_stats.txt', 'a', newline='') as f:
            writer = csv.writer(f)
            for stat in all_stats:
                writer.writerow([stat])
