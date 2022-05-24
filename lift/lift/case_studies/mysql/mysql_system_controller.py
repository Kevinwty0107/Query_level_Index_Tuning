import sys
from copy import deepcopy

from lift.case_studies.common.index_net import build_index_net
from lift.case_studies.mysql.mysql_converter import MySQLConverter
from lift.case_studies.mysql.mysql_data_source import MySQLDataSource
from lift.case_studies.mysql.mysql_schema import MySQLSchema
from lift.case_studies.mysql.mysql_system_environment import MySQLSystemEnvironment
from lift.case_studies.mysql.tpch_synthetic_workload import TPCHSyntheticWorkload
from lift.case_studies.mysql.tpch_util import SCALE_FACTOR
from lift.controller.system_controller import SystemController

import time
import numpy as np
import csv

from lift.rl_model.task import Task
from lift.rl_model.task_graph import TaskGraph
from lift.util.parsing_util import sort_input


class MySQLSystemController(SystemController):
    """
    MySQL online controller - executes a set of training or test queries for
    a number of specified steps on a MySQL instance and records the result.
    """

    def __init__(
        self,
        agent_config,
        network_config,
        experiment_config,
        schema_config,
        result_dir="",
        demo_data_label="",
        model_store_path="",
        model_load_path="",
        blackbox_optimization=False
    ):
        super(MySQLSystemController, self).__init__(
            agent_config=agent_config,
            experiment_config=experiment_config,
            network_config=network_config,
            result_dir=result_dir,
            model_store_path=model_store_path,
            model_load_path=model_load_path,
        )
        self.schema_config = schema_config
        self.demo_data_label = demo_data_label

        # Optimization mode.
        self.blackbox_optimization = blackbox_optimization

        tables = schema_config.get("tables", None)
        if tables is None:
            tables = ["ORDERS", "LINEITEM", "PART"]
            schema_config["tables"] = tables
        self.schema = MySQLSchema(
            schema_config=schema_config,
            mode=self.state_mode
        )

        self.states_spec = self.schema.states_spec
        self.actions_spec = self.schema.actions_spec
        vocab_size = len(self.schema.get_system_spec()['input_vocabulary'].keys())
        layer_size = experiment_config['layer_size']

        agent_config["network_spec"] = build_index_net(
                state_mode=self.state_mode,
                states_spec=self.states_spec,
                embed_dim=experiment_config["embedding_size"],
                vocab_size=vocab_size,
                layer_size=layer_size)

        # Converters states and actions.
        self.converter = MySQLConverter(
            schema=self.schema,
            experiment_config=experiment_config
        )

        # Workload.
        num_selections = experiment_config.get("num_selections", 3)
        # Only used for black-box search.
        self.max_reward = 0
        self.best_train_index_set = []

        # Workload spec.
        self.num_executions = experiment_config['num_executions']
        self.training_episodes = experiment_config["training_episodes"]
        self.queries_per_episode = experiment_config["queries_per_episode"]

        # On how many different episodes do we evaluate the trained model?
        self.test_episodes = experiment_config["test_episodes"]

        workload_spec = {
            "tables": tables,
            "scale_factor": SCALE_FACTOR,
            "num_selections": num_selections
        }
        self.workload = TPCHSyntheticWorkload(workload_spec=workload_spec)

        # Executes changes on system.
        self.system_environment = MySQLSystemEnvironment(
            all_tables=tables
        )

        # Loads trace data.
        self.data_source = MySQLDataSource(
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
        # In blackbox mode, single set.
        if self.blackbox_optimization:
            num_train_queries = self.queries_per_episode
            train_queries = [self.workload.generate_query_template() for _ in range(num_train_queries)]
            test_queries = [deepcopy(q) for q in train_queries]
        else:
            num_train_queries = self.training_episodes * self.queries_per_episode
            train_queries = [self.workload.generate_query_template() for _ in range(num_train_queries)]

            num_test_queries = self.test_episodes * self.queries_per_episode
            test_queries =[self.workload.generate_query_template() for _ in range(num_test_queries)]

        if export:
            self.data_source.export_data(data=train_queries, data_dir=self.result_dir, label=label + "_train")
            self.data_source.export_data(data=test_queries, data_dir=self.result_dir, label=label + "_test")

        return train_queries, test_queries

    def train(self, queries, label):
        """
        Executes training and exports results.

        Args:
            queries (list): List of train queries. Note that these are all queries across
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
        self.time_step = 0

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
                action = self.task_graph.act_task("", states=state.get_value(), apply_preprocessing=True,
                                                  use_exploration=True, time_percentage=self.time_step / self.max_steps)
                system_action = self.converter.agent_to_system_action(
                    actions=action,
                    meta_data=dict(query_columns=query.query_columns)
                )
                acting_time = time.monotonic() - interaction_start
                system_runtime = self.system_environment.act(dict(index=system_action, table=query.query_table))
                episode_system_waiting_time += system_runtime

                noop_action = self.system_environment.is_noop(system_action)
                runtime = self.get_training_runtimes(episode_queries, query, noop_action)
                index_size, num_indices, final_index = self.system_environment.system_status()

                update_start = time.monotonic()
                reward = self.converter.system_to_agent_reward(meta_data=dict(
                    runtime=runtime, index_size=index_size))
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

                episode_reward += reward  # We could do this differently by measuring reward at the end..
                runtimes.append(runtime)
                total_size = index_size
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

    def get_training_runtimes(self, episode_queries, query, noop=False):
        if self.training_reward == 'full':
            query_runtimes = []
            for ep_query in episode_queries:
                query_string, query_args = ep_query.sample_query()
                query_runtimes.append(self.system_environment.execute(query_string, query_args))
            runtime = self.runtime_aggregation(query_runtimes)
        elif self.training_reward == 'differential':
            query_runtimes = []
            for i, ep_query in enumerate(episode_queries):
                query_string, query_args = ep_query.sample_query()
                if str(ep_query.query_string) in self.runtime_cache:
                    # Common columns -> re-execute
                    # Only if action was not noop -> noop action should mean no change.
                    if not noop and set(ep_query.query_columns).intersection(set(query.query_columns)):
                        # self.logger.info('Runtime cached but intersecting columns ep query = {}, query = {}'.
                        #                  format(ep_query.query_columns, query.query_columns))
                        t = self.system_environment.execute(query_string, query_args)
                        # Cache runtime of that query.
                        self.runtime_cache[str(ep_query.query_string)] = t
                        query_runtimes.append(t)
                    else:
                        # self.logger.info('Using cached runtime for episode query = {}'.format(i))
                        # No common columns -> use  cached value.
                        query_runtimes.append(self.runtime_cache[str(ep_query.query_string)])
                else:
                    # self.logger.info('No cache entry for episode query = {}'.format(i))
                    t = self.system_environment.execute(query_string, query_args)
                    # Cache runtime of that query.
                    self.runtime_cache[str(ep_query.query_string)] = t
                    query_runtimes.append(t)
            runtime = self.runtime_aggregation(query_runtimes)
        elif self.training_reward == 'differential_uncached':
            query_runtimes = []
            for ep_query in episode_queries:
                query_string, query_args = ep_query.sample_query()
                query_runtimes.append(self.system_environment.execute(query_string, query_args))
            runtime = self.runtime_aggregation(query_runtimes)
        else:
            query_string, query_args = query.sample_query()
            runtime = self.system_environment.execute(query_string, query_args)
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
        context = []
        system_actions = []
        test_queries = sort_input(test_queries)
        if demo_rule is None:
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
                self.system_environment.act(dict(index=system_action, table=query.query_table))
                system_actions.append(system_action)
                context.append(system_action)
        elif index_set is not None:
            self.logger.info("Acting with provided index set.")
            # Assume query set on same table here.
            table = test_queries[0].query_table
            for index_action in index_set:
                self.system_environment.act(dict(index=index_action, table=table))
                system_actions.append(index_action)
        else:
            self.logger.info("Evaluating with demo rule.")
            # Create all indices according to demo rule.
            for query in test_queries:
                system_action = demo_rule.generate_demonstration(states=query, context=context)
                self.system_environment.act(system_action)
                context.append(system_action)

        return system_actions

    def restore_workload(self, train_label, test_label):
        """
        Restores a workload from serialised version.

        """
        train_queries = self.data_source.load_data(self.result_dir, label=train_label)
        test_queries = self.data_source.load_data(self.result_dir, label=test_label)

        return train_queries, test_queries

    def evaluate(self, queries, label, actions=None, export_evaluation=True):
        """
        Evaluates a set of queries. This assumes indices have already been created by whatever means
        (rule, pretrained model, online trained model, etc.).

        Args:
            queries (list): List of queries to execute.
            label (str): Label for this evaluation. Results will be saved under this label.
            actions (list): Optional actions used for this evaluation. Will be jointly exported.
            export_evaluation (bool):
        """
        runtimes = []
        for query in queries:
            runtime = []
            for _ in range(self.num_executions):
                query_string, query_args = query.sample_query()
                exec_time = self.system_environment.execute(query_string, query_args)
                runtime.append(exec_time)
            runtimes.append((np.mean(runtime), np.std(runtime)))

        index_size, num_indices, final_index = self.system_environment.system_status()
        if export_evaluation:
            self.export(label=label, runtimes=runtimes, index_size=index_size, num_indices=num_indices,
                        final_index=final_index, queries=queries, indices=actions)

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
                writer.writerow([runtime_tuple[0], runtime_tuple[1], query, index])

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
