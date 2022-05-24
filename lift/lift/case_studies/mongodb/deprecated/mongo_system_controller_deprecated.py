from rlgraph.agents import Agent

from lift.controller.system_controller import SystemController
from lift.case_studies.mongodb import mongo_model_generators, mongo_schemas, MongoDataSource
from lift.case_studies.mongodb.mongodb_system_environment import MongoDBSystemEnvironment
import time
import numpy as np
import csv
from lift.util.parsing_util import query_priority


class MongoSystemController(SystemController):
    """
    MongoDB online controller - executes a set of training or test queries for
    a number of specified steps on a MongoDB instance and records the result.
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
        train_label=None,
        test_label=None,
        model_store_path="",
        model_load_path="",
        store_model=False,
        load_model=False,
        training_dir=None,
        test_dir=None
    ):
        super(MongoSystemController, self).__init__(
            agent_config=agent_config,
            experiment_config=experiment_config,
            network_config=network_config,
            result_dir=result_dir,
            model_store_path=model_store_path,
            model_load_path=model_load_path,
            store_model=store_model,
            load_model=load_model
        )
        self.schema_config = schema_config
        self.demo_data_label = demo_data_label
        self.train_label = train_label
        self.test_label = test_label
        self.training_dir = training_dir
        self.test_dir = test_dir

        self.training_baseline_label = experiment_config['default_baseline_name']
        self.extra_baseline_label = experiment_config['extra_baseline_name']

        # How often to execute each query.
        self.num_executions = experiment_config['num_executions']
        self.training_steps = experiment_config['training_steps']
        self.train_train_queries = experiment_config['train_train_queries']
        self.train_test_queries = experiment_config['train_test_queries']
        self.rewards_available = experiment_config.get('rewards_available', False)
        self.sample_query_values = experiment_config.get('sample_values', False)

        self.evaluate_pretrain = experiment_config.get('evaluate_pretrain', False)
        self.evaluate_default = experiment_config.get('evaluate_default', False)

        self.schema = mongo_schemas[experiment_config['model']](
            schema_config=schema_config,
            experiment_config=experiment_config
        )

        self.states_spec = self.schema.states_spec
        self.actions_spec = self.schema.actions_spec
        vocab_size = len(self.schema.get_system_spec()['input_vocabulary'].keys())
        layer_size = experiment_config['layer_size']

        if network_config:
            network_spec = network_config
        else:
            network_spec = [
                dict(type='embedding', embed_dim=layer_size, vocab_size=vocab_size),
                dict(type="reshape", flatten=True),
                dict(type='dense', units=layer_size, activation='relu', scope="dense_1")
            ]
        agent_config["network_spec"] = network_spec

        # Converters states and actions.
        self.converter = mongo_model_generators[experiment_config['model']](
            schema=self.schema,
            experiment_config=experiment_config
        )

        # Executes changes on system.
        self.system_environment = MongoDBSystemEnvironment(
            experiment_config=experiment_config,
            schema=self.schema,
            host=host,
            converter=self.converter
        )

        # Loads trace data.
        self.data_source = MongoDataSource(
            converter=self.converter,
            schema=self.schema
        )

        # Creates RLgraph agent from spec.
        self.agent = Agent.from_spec(
            self.agent_config,
            state_space=self.states_spec,
            action_space=self.actions_spec
        )
        # N.b. there was a finalize on default graph here.

        # Compare reward to max reward, save best index set found so far
        # If true, compare to best found, if false, compare to final
        self.compare_to_best = self.experiment_config.get('compare_to_best', False)
        self.max_reward = -1000000
        self.best_train_index_set = []

    def init_workload(self, *args, **kwargs):
        train_queries = self.load_queries(query_dir=self.training_dir, label=self.train_label)
        test_queries = self.load_queries(query_dir=self.test_dir, label=self.test_label)
        train_queries = self.sort_input(train_queries)
        test_queries = self.sort_input(test_queries)

        return train_queries, test_queries

    def run(self):
        train_queries = self.load_queries(query_dir=self.training_dir, label=self.train_label)
        test_queries = self.load_queries(query_dir=self.test_dir, label=self.test_label)

        # Evaluate train and test no-index performance.
        self.system_environment.reset()
        self.logger.info("Evaluating default baseline (no indexing).")
        if self.train_train_queries:
            runtimes = self.execute_queries(queries=train_queries, runs=self.num_executions)
            index_size, num_indices, final_index = self.system_environment.system_status()
            self.export(self.train_label + 'default_results', runtimes, index_size, num_indices, final_index, train_queries)
        if self.train_test_queries or self.evaluate_default:
            runtimes = self.execute_queries(queries=test_queries, runs=self.num_executions)
            index_size, num_indices, final_index = self.system_environment.system_status()
            self.export(self.test_label + 'default_results', runtimes, index_size, num_indices, final_index, test_queries)

        if self.experiment_config['load_indices']:
            self.logger.info("Evaluating training baseline.")
            if self.train_train_queries:
                self.evaluate_baseline(queries=train_queries, file_label=self.train_label,
                                       baseline_label=self.training_baseline_label,
                                       label=self.train_label + 'baseline')
            # Evaluate default baseline
            self.evaluate_baseline(queries=test_queries, file_label=self.test_label,
                                   baseline_label=self.training_baseline_label,
                                   label=self.test_label + self.training_baseline_label + 'baseline')
            # Evaluate extra baseline
            self.evaluate_baseline(queries=test_queries, file_label=self.test_label,
                                   baseline_label=self.extra_baseline_label,
                                   label=self.test_label + self.extra_baseline_label + 'baseline')
        train_queries = self.sort_input(train_queries)
        test_queries = self.sort_input(test_queries)
        if self.load_model:
            self.logger.info("Loading model from: {}.".format(self.model_load_path))
            self.agent.load_model(checkpoint_directory=self.model_load_path)
            self.logger.info("Evaluating pretrained model.")

            if self.train_train_queries:
                actions = self._act_on_query_set(train_queries)
                runtimes = self.execute_queries(queries=train_queries, runs=self.num_executions)
                index_size, num_indices, final_index = self.system_environment.system_status()
                self.export(self.train_label + 'pretrain_results', runtimes, index_size,
                            num_indices, final_index, train_queries, actions)
                self.system_environment.reset()
            if self.train_test_queries or self.evaluate_pretrain:
                self.system_environment.reset()
                actions = self._act_on_query_set(test_queries)
                runtimes = self.execute_queries(queries=test_queries, runs=self.num_executions)
                index_size, num_indices, final_index = self.system_environment.system_status()
                self.export(self.test_label + 'pretrain_results', runtimes, index_size,
                            num_indices, final_index, test_queries, actions)
                self.system_environment.reset()

        if self.experiment_config["use_demo_data_online"]:
            # If we want to continue updating from demo data throughout online training.
            batch = self.data_source.load_data(self.training_dir, label=self.demo_data_label,
                                               baseline_label=self.training_baseline_label, rewards_available=self.rewards_available)
            self.agent.observe_demos(prerocessed_states=batch["states"], actions=batch["actions"],
                rewards=batch["rewards"], next_states=batch["next_states"], terminals=batch["terminals"])

        if self.train_train_queries:
            self.logger.info("Beginning training loop on train queries.")
            self.system_environment.reset()
            self.train(train_queries, self.training_steps, mode='')
            self.agent.store_model(self.model_store_path, False)
            self.system_environment.reset()
        if self.train_test_queries:
            self.logger.info("Beginning training loop on test queries.")
            self.system_environment.reset()
            self.train(test_queries, self.training_steps, mode='')
            self.agent.store_model(self.model_store_path, False)
            self.system_environment.reset()
        # Generate and evaluate final model on train queries.

        if self.train_train_queries:
            self.logger.info("Performing final evaluation on train queries, exporting results.")
            if self.compare_to_best:
                actions = self.best_train_index_set
                self.create_indices(actions)
            else:
                actions = self._act_on_query_set(train_queries)

            runtimes = self.execute_queries(queries=train_queries, runs=self.num_executions)
            index_size, num_indices, final_index = self.system_environment.system_status()
            self.export(self.train_label + 'final_online_results', runtimes, index_size,
                        num_indices, final_index, train_queries, actions)
            self.system_environment.reset()

        # Generate and evaluate final model on test data.
        if self.train_test_queries:
            self.logger.info("Performing final evaluation on test queries, exporting results.")
            if self.compare_to_best:
                actions = self.best_train_index_set
                self.create_indices(actions)
            else:
                actions = self._act_on_query_set(train_queries)
            runtimes = self.execute_queries(queries=test_queries, runs=self.num_executions)
            index_size, num_indices, final_index = self.system_environment.system_status()
            self.export(self.test_label + 'final_online_results', runtimes, index_size,
                        num_indices, final_index, test_queries, actions)
            self.system_environment.reset()

    def run_online_only(self):
        train_queries = self.load_queries(query_dir=self.training_dir, label=self.train_label)
        test_queries = self.load_queries(query_dir=self.test_dir, label=self.test_label)
        train_queries = self.sort_input(train_queries)
        test_queries = self.sort_input(test_queries)

        # Evaluate train and test no-index performance.
        self.system_environment.reset()
        if self.train_train_queries:
            self.logger.info("Beginning training loop on train queries.")
            self.system_environment.reset()
            self.train(train_queries, self.training_steps, mode='online_only_')
            self.system_environment.reset()
        if self.train_test_queries:
            self.logger.info("Beginning training loop on test queries.")
            self.system_environment.reset()
            self.train(test_queries, self.training_steps, mode='online_only_')
            self.system_environment.reset()

        # Generate and evaluate final model on test data.
        self.logger.info("Performing final evaluation on test queries, exporting results.")
        actions = self._act_on_query_set(test_queries)
        runtimes = self.execute_queries(queries=test_queries, runs=self.num_executions)
        index_size, num_indices, final_index = self.system_environment.system_status()
        self.export(self.test_label + 'online_only_final_online_results', runtimes, index_size,
                    num_indices, final_index, test_queries, actions)
        self.system_environment.reset()

    def evaluate_tf_model(self, path, label="", **kwargs):
        test_queries = self.load_queries(query_dir=self.test_dir, label=self.test_label)
        test_queries = self.sort_input(test_queries)
        self.agent.load_model(checkpoint_directory=path)

        # Generate and evaluate final model on test data.
        self.logger.info("Performing final evaluation on test queries, exporting results.")
        actions = self._act_on_query_set(test_queries)
        runtimes = self.execute_queries(queries=test_queries, runs=self.num_executions)
        index_size, num_indices, final_index = self.system_environment.system_status()
        if label:
            file = self.test_label + label
        else:
            file = self.test_label + 'online_only_final_online_results'

        self.export(file, runtimes, index_size, num_indices, final_index, test_queries, actions)
        self.system_environment.reset()

    def evaluate_indices_from_file(self, path, index_label):
        self.system_environment.reset()
        parsed_indices = self.data_source.parse_indices_to_actions(path)
        queries = self.load_queries(query_dir=self.test_dir, label=self.test_label)
        self.logger.info("Creating all baseline indices:")
        for index_action in parsed_indices:
            self.system_environment.act(index_action)
        index_size, num_indices, final_index = self.system_environment.system_status()
        self.logger.info("Baseline indices created = {}".format(num_indices))
        runtimes = self.execute_queries(queries, runs=self.num_executions)

        label = self.test_label + index_label + 'baseline'
        self.export(label, runtimes, index_size, num_indices, final_index, queries, parsed_indices)

    def evaluate_baseline(self, queries, file_label, baseline_label, label):
        """
        Evaluates a set of indices, e.g. provided by a human expert or rules.
        """
        self.system_environment.reset()
        index_path = self.training_dir + '/' + file_label + baseline_label + 'indices.csv'
        parsed_indices = self.data_source.parse_indices_to_actions(index_path)

        self.logger.info("Creating all baseline indices:")
        for index_action in parsed_indices:
            self.system_environment.act(index_action)
        index_size, num_indices, final_index = self.system_environment.system_status()
        self.logger.info("Baseline indices created = {}".format(num_indices))
        runtimes = self.execute_queries(queries, runs=self.num_executions)
        self.export(label, runtimes, index_size, num_indices, final_index, queries, parsed_indices)

    def train(self, queries, episodes, mode=''):
        """
        Executes training for a specified number of iterations.
        :param queries: Queries to train on
        :param episodes: Training steps
        """
        last_index = len(queries) - 1
        episode_rewards = []
        episode_times = []
        # Time spent waiting on action execution.
        system_waiting_times = []
        # Time spent interacting with agent via parsing, act() and observe().
        agent_interaction_times = []
        evaluation_times = []

        for i in range(episodes):
            self.agent.reset()
            self.logger.info("Starting training episode {}.".format(i))
            episode_start = time.monotonic()
            self.system_environment.reset()
            context = []
            query_index = 0
            episode_reward = 0
            episode_system_waiting_time = 0
            episode_agent_interaction_time = 0
            runtimes = []
            total_size = 0

            for query in queries:
                interaction_start = time.monotonic()
                state = self.converter.system_to_agent_state(query.get_query_dict()['query_filter'],
                                                             system_context=dict(
                        sort_info=query.get_query_dict()['sort_order'],
                        index_field_list=context,
                        aggregation=query.get_query_dict()['aggregation']
                    )
                                                             )

                state_input = state.get_value()
                action = self.agent.get_action(state_input, use_exploration=True)
                system_action = self.converter.agent_to_system_action(
                    action,
                    meta_data=dict(query_columns=state.get_meta_data()['query_columns'])
                )
                acting_time = time.monotonic() - interaction_start
                system_runtime = self.system_environment.act(system_action)
                episode_system_waiting_time += system_runtime
                start = time.monotonic()
                query.execute()
                runtime = time.monotonic() - start
                index_size, num_indices, final_index = self.system_environment.system_status()

                update_start = time.monotonic()
                reward = self.converter.system_to_agent_reward(meta_data=dict(
                    runtime=runtime, index_size=index_size))

                query_index += 1
                terminal = query_index == last_index
                context.append(system_action)
                # Last entry: next state = state
                next_state_index = query_index if query_index <= last_index else query_index - 1
                # Get next state from list of queries and combine with context.
                next_q = queries[next_state_index]

                next_state = self.converter.system_to_agent_state(
                    query_filter=next_q.get_query_dict()['query_filter'],
                    system_context=dict(sort_info=next_q.get_query_dict()['sort_order'], index_field_list=context,
                                        aggregation=next_q.get_query_dict()['aggregation'])
                )
                self.agent.observe(state_input, action, [], reward, next_state.get_value(), terminal)

                update_time = time.monotonic() - update_start
                episode_agent_interaction_time += (acting_time + update_time)

                episode_reward += reward  # We could do this differently by measuring reward at the end..
                runtimes.append(runtime)
                total_size = index_size

            if episode_reward > self.max_reward:
                self.max_reward = episode_reward
                # Context are the indices created thus far.
                self.best_train_index_set = context

            episode_times.append(time.monotonic() - episode_start)
            self.logger.info("Finished episode: episode reward: {}, mean runtime: {}, total size: {}.".format(
                episode_reward, np.mean(runtimes), total_size))
            episode_rewards.append(episode_reward)
            system_waiting_times.append(episode_system_waiting_time)
            agent_interaction_times.append(episode_agent_interaction_time)
            evaluation_times.append(np.sum(runtimes))

        # Export episode rewards and timings.
        np.savetxt(self.result_dir + '/' + mode + 'episode_rewards.txt', np.asarray(episode_rewards), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + mode + 'episode_durations.txt', np.asarray(episode_times), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + mode + 'system_waiting_times.txt', np.asarray(system_waiting_times), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + mode + 'agent_action_times.txt', np.asarray(agent_interaction_times), delimiter=',')
        np.savetxt(self.result_dir + '/timing/' + mode + 'evaluation_times.txt', np.asarray(evaluation_times), delimiter=',')

    def execute_queries(self, queries, runs):
        """
        Executes a set of queries, to be used for final evaluation once indexing is complete.
        :param queries: Queries
        :param num_executions: How often to execute each query
        :return: List of tuples of  Mean and std of runtime per query.
        """
        runtimes = []
        for query in queries:
            runtime = []
            for _ in range(runs):
                exec_time = query.execute()
                runtime.append(exec_time)
            runtimes.append((np.mean(runtime), np.std(runtime)))
        return runtimes

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

    def _act_on_query_set(self, queries):
        """
        Acts once by cycling through a set of queries and performing the corresponding actions
        to create a final system configuration.
        """
        context = []
        system_actions = []
        for query in queries:
            state = self.converter.system_to_agent_state(query.get_query_dict()['query_filter'], system_context=dict(
               sort_info=query.get_query_dict()['sort_order'],
               index_field_list=context,
               aggregation=query.get_query_dict()['aggregation']
            ))
            action = self.agent.get_action(states=state.get_value(), apply_preprocessing=True)
            system_action = self.converter.agent_to_system_action(
                actions=action,
                meta_data=dict(query_columns=state.get_meta_data()['query_columns'])
            )
            self.system_environment.act(action=system_action)
            system_actions.append(system_action)
            context.append(system_action)

        return system_actions

    def load_queries(self, query_dir, label):
        path = str(query_dir) + '/' + str(label) + 'queries.json'
        self.logger.info("Loading queries from path = {}".format(path))
        train_queries = MongoDataSource.load_query_dicts(query_path=path)

        return self.system_environment.make_executable(
            queries=train_queries,
            sample_values=self.sample_query_values
        )

    def sort_input(self, queries):
        for query in queries:
            filter_dict = query.get_query_dict()
            priority = query_priority(filter_dict, list(self.schema.schema_config.keys()))
            query.priority = priority
        return sorted(queries, key=lambda query: query.priority, reverse=True)

    def create_indices(self, indices):
        # Just create a set of indices.
        for index_action in indices:
            self.system_environment.act(index_action)