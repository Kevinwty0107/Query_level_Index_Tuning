from lift.baselines.gpy_opt import GPyOpt
import logging
import time
import numpy as np

from lift.case_studies.mongodb import mongo_model_generators
from lift.case_studies.common.baseline_util import export

try:
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    print("Could not import GPyOpt!")


class GPyOptMongoDB(GPyOpt):
    """
    Interfaces the Bayesian optimisation library GPyOpt:

    https://github.com/SheffieldML/GPyOpt

    API instructions, examples at:

    http://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_reference_manual.ipynb
    """

    def __init__(self, queries, experiment_config, schema, result_dir, system_model):
        self.logger = logging.getLogger(__name__)
        self.queries = queries
        self.opt = self.build_tasks()
        # Map name of query to actions.
        self.action_dict = {}
        # Map name of query to query object.
        self.query_name_dict = {}
        self.model_generator = mongo_model_generators[experiment_config['model']](
            schema=schema,
            experiment_config=experiment_config
        )
        super(GPyOptMongoDB, self).__init__(experiment_config, schema, result_dir, system_model)
        self.logger.info("Initializing OpenTuner task descriptions.")

    def act(self, states, *args, **kwargs):
        # TODO do actions arrive as named dict or as list of values?
        self.logger.info("Input actions for iteration:", states)
        cfg = states
        context = []
        self.system_model.reset()

        episode_reward = 0
        for query_name, query_actions in self.action_dict.items():
            # Fetch each configuration for that action.
            action = {}
            for action_name in query_actions:
                action_value = (cfg[action_name])
                action_only_name = action_name.split("+")[1]
                action[action_only_name] = action_value
            self.logger.info("Executing for query name {} with action {}".format(query_name, action))

            query_obj = self.query_name_dict[query_name]
            state = self.model_generator.system_to_agent_state(query_obj.get_query_dict()['query_filter'], system_context=dict(
                    sort_info=query_obj.get_query_dict()['sort_order'],
                    index_field_list=context,
                    aggregation=query_obj.get_query_dict()['aggregation']
            ))
            system_action = self.model_generator.agent_to_system_action(
                actions=action,
                meta_data=dict(query_fields=state.get_meta_data()['query_fields'])
            )
            self.system_model.act(system_action)
            context.append(system_action)

            start = time.monotonic()
            query_obj.execute()
            runtime = time.monotonic() - start
            index_size, num_indices, final_index = self.system_model.system_status()

            reward = self.model_generator.system_to_agent_reward(meta_data=dict(
                runtime=runtime, index_size=index_size))
            episode_reward += reward
        return episode_reward

    def build_tasks(self):
        actions = []
        domain = []
        num_outputs = self.schema.get_system_spec()["num_outputs"]

        # Create one parameter for each action to take per query.
        for i, query in enumerate(self.queries):
            query_name = "q_{}".format(i)
            # Define 3 actions for each query, save names.
            action_list = []
            for k in range(3):
                action_name = "index_field{}".format(k)
                prefixed_name = "{}+{}".format(query_name, action_name)

                # Min and max are inclusive.
                action_list.append(prefixed_name)
                # Prefix with query or we try to create the same parameter many times.

                # GPyOpt action format.
                domain.append(
                    {'name': prefixed_name, 'type': 'discrete', 'domain': (0, num_outputs - 1)}
                )

            # Save query to its actions to identify later.
            self.action_dict[query_name] = action_list
            self.query_name_dict[query_name] = query

        # We will model this by presenting bayesopt with a concatenation of all query states
        # into one input.
        return GPyOpt.methods.BayesianOptimization(f=self.act,  # Returns reward
                                                   domain=actions,  # box-constraints of the problem
                                                   acquisition_type='EI',
                                                   verbosity=True,
                                                   exact_feval=True)

    def eval_best(self, config, label):
        # Create final configuration in system.
        ep_reward, actions = self.act(config)

        # Do a final eval of queries.
        runs = self.config['num_executions']
        runtimes = []
        for query in self.queries:
            runtime = []
            for _ in range(runs):
                exec_time = query.execute()
                runtime.append(exec_time)
            runtimes.append((np.mean(runtime), np.std(runtime)))

        # Export results.
        index_size, num_indices, final_index = self.system_model.system_status()
        self.logger.info("Exporting final results.")
        export(self.result_dir, label, runtimes, index_size, num_indices, final_index, self.queries, actions)
