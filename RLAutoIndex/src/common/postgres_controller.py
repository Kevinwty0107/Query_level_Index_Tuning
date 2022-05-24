#!/usr/local/bin/python3

import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.controller.system_controller import SystemController

sys.path.insert(0, os.path.join(head, '..')) # src
from dqn_agent.postgres_converter import PostgresConverter as PostgresDQNConverter
from dqn_agent.postgres_schema import PostgresSchema as PostgresDQNSchema
from spg_agent.postgres_converter import PostgresConverter as PostgresSPGConverter
from spg_agent.postgres_schema import PostgresSchema as PostgresSPGSchema


from postgres_agent import PostgresAgent
from postgres_data_source import PostgresDataSource
from postgres_system_environment import Action, PostgresSystemEnvironment
from tpch_workload import TPCHWorkload, sort_queries
from tpch_util import tpch_table_columns

import copy
import csv
import itertools
import numpy as np
import pdb
import pickle
import time
from tensorboardX import SummaryWriter

class PostgresSystemController(SystemController):
    """
    Abstraction for training + testing 
    
    Encapsulates a slew of other abstractions: agent + system + agent-system auxiliaries like converter and schema
    
    Some of these are shared by DQN + SPG agents, while some are not shared, so there's a bit of brittleness in how this is architected,
    though should be fairly straightforward to follow.

    """

    def __init__(self, 
                 dqn,
                 blackbox,
                 agent_config, 
                 experiment_config, 
                 schema_config,
                 result_dir):
        """
        
        Args:
            dqn (bool): DQN agent, else SPG agent; required where rlgraph breaks abstraction barrier (e.g. resetting)
            ...

             TODO 'config' and 'spec' conflated throughout code
        """
        self.dqn = dqn
        self.blackbox = blackbox

        super().__init__(agent_config=agent_config, 
                         experiment_config=experiment_config,
                         result_dir=result_dir)    

        #
        # schema, for agent state and action representations 
        # 
        schema_config['tables'] = experiment_config['tables']
        self.schema_config = schema_config
        
        if self.dqn:
            self.schema = PostgresDQNSchema(schema_config)
        else:
            self.schema = PostgresSPGSchema(schema_config)

        self.states_spec = self.schema.get_states_spec()
        self.actions_spec = self.schema.get_actions_spec()
        self.system_spec = self.schema.get_system_spec()

        if self.dqn: self.agent_config['network_spec'][0]['vocab_size'] = self.system_spec['vocab_size'] # TODO

        #
        # converter
        #
        if self.dqn:
            self.converter = PostgresDQNConverter(experiment_config=experiment_config, schema=self.schema)
        else:
            self.converter = PostgresSPGConverter(experiment_config=experiment_config, schema=self.schema)

        #
        # workload
        # 
        n_selections = experiment_config.get("n_selections", 3)
                
        self.n_executions = experiment_config['n_executions']
        self.n_train_episodes = experiment_config["n_train_episodes"]
        self.n_test_episodes = experiment_config["n_test_episodes"]
        self.n_queries_per_episode = experiment_config["n_queries_per_episode"]

        workload_spec = {
            "tables": self.schema_config['tables'],
            "scale_factor": 1 , # TODO specify this somewhere, not just in tpch_util 
            "n_selections": n_selections
        }
        self.workload = TPCHWorkload(spec=workload_spec)

        #
        # workload ser/des
        #
        self.data_source = PostgresDataSource(workload_spec)

        #
        # environment
        #
        self.system_environment = PostgresSystemEnvironment(tbls=self.experiment_config['tables'])
        
        self.logger.info('computing column selectivities...')
        start = time.monotonic()
        try:
            with open(os.path.join(result_dir, 'selectivities.pkl'), 'rb') as f:
                self.system_environment.tbl_2_col_2_sel = pickle.load(f)
        except:
            self.system_environment.compute_column_selectivity()
            with open(os.path.join(result_dir, 'selectivities.pkl'), 'wb') as f:
                pickle.dump(self.system_environment.tbl_2_col_2_sel, f)
        self.logger.info('...took {:.2f} seconds'.format(time.monotonic() - start))

        #
        # agent
        #
        self.agent_api = PostgresAgent(
            dqn=self.dqn,
            agent_config=self.agent_config,
            experiment_config=self.experiment_config,
            schema = self.schema
        )
        
        if self.dqn:
            self.task_graph = self.agent_api.task_graph # have to have this exposed...
            self.set_update_schedule(self.agent_config["update_spec"])

            # TODO:
            # per description in docstring, we can't hide all rlgraph agent behind a shared dqn/spg api:
            # self.set_update_schedule(), self.agent.timesteps, self.update_if_necessary() called from super class 
            # require access to rlpgraph *directly*


    def generate_workload(self, export=False):
        """
        """
        if self.blackbox:
            n_train_queries = self.n_queries_per_episode
            train_queries = [self.workload.generate_query_template(selectivities=self.system_environment.tbl_2_col_2_sel) 
                             for _ in range(n_train_queries)]
            test_queries = [copy.deepcopy(train_query) for train_query in train_queries]
        else: 
            n_train_queries = self.n_train_episodes * self.n_queries_per_episode
            n_test_queries = self.n_test_episodes * self.n_queries_per_episode
            train_queries = [self.workload.generate_query_template(selectivities=self.system_environment.tbl_2_col_2_sel) 
                             for _ in range(n_train_queries)]
            test_queries = [self.workload.generate_query_template(selectivities=self.system_environment.tbl_2_col_2_sel) 
                            for _ in range(n_test_queries)]

        # some really rough statistics 
        # selectivity -- are these good candidates for indices?
        total = 0.0
        cnt = 0
        for query in train_queries:
            for query_col in query.query_cols:
                cnt += 1
                total += self.system_environment.tbl_2_col_2_sel[query.query_tbl][query_col]
        self.logger.info('avg col selectivity of cols in training queries: {}'.format(total / float(cnt)))

        # intersection -- are these good candidates for compound indices?
        if self.blackbox:
            intxns = 0
            for query in train_queries: 
                intxns_per_query = 0
                for other_query in query: 
                    if list(query.query_cols) == list(other_query.query_cols[:len(query.query_cols)]):
                        intxns_per_query += 1 
                    if intxns_per_query > 1: 
                        intxns += 1 
                        break
            self.logger.info('% of queries as candidates for intersection: {}'.format(intxns / n_train_queries))
        else:
            intxn_cnts = []
            for i in range(self.n_train_episodes):
                queries = train_queries[i * self.n_queries_per_episode:(i+1) * self.n_queries_per_episode]
                intxns = 0 
                for query in queries:
                    intxns_per_query = 0
                    for other_query in queries: 
                        if list(query.query_cols) == list(other_query.query_cols[:len(query.query_cols)]):
                            # if there's an opportunity for intersection, assuming full indices
                            # e.g. if [foo, bar] == [foo, bar, baz][:2] 
                            intxns_per_query += 1 
                        if intxns_per_query > 1: 
                            # > 1 b/c query will intersect with itself
                            intxns += 1 
                            break 
                intxn_cnts.append(intxns)             
            self.logger.info('avg % of queries as candidates for intersection for each episode: {}'.format( (np.array(intxn_cnts) / self.n_queries_per_episode).mean()))

        if export:
            self.data_source.export_data(train_queries, self.result_dir, label='train')
            self.data_source.export_data(test_queries, self.result_dir, label='test')

        return train_queries, test_queries
    
    def restore_workload(self, path=None):
        """
            path (str): path to a workload to rerun an agent on, added ad hoc
        """
        train_queries = self.data_source.import_data(self.result_dir, label="train")
        test_queries = self.data_source.import_data(self.result_dir, label="test", path=path)

        return train_queries, test_queries

    def train(self, queries):
        """
            trains in same MDP-style as 3.1 in Sutton-Barto

            Args:
                queries (list): SQLQueries, each episode exposes a slice of queries (query templates) 
                label (str): indicates whether spg or dqn

            TODO:
                recording of meta-data for an episode / episodes reduces code readability, not sure if there's a clean workaround
        """

        # meta-data
        result_dir = os.path.join(self.result_dir, 'dqn' if self.dqn else 'spg')
        if not os.path.exists(result_dir): os.makedirs(result_dir)

        tensorboard = SummaryWriter(os.path.join(result_dir, 'tensorboard'))
        step = 0

        # record per step 
        times = [] # total time, roughly decomposes into...
        agent_times = [] # time spent retrieving action, updating based on retrieved action
        query_times = [] # time spent on queries
        system_times = [] # time spent on transitioning state (i.e. indexing)
        rewards = [] # for trend over time
        selectivities = []
        intersections = []
        losses = []

        # record per episode of steps
        index_set_stats = [] # distribution of action decisions over steps, of # of query attributes per step, the total size

        if self.dqn: self.task_graph.get_task("").unwrap().timesteps = 0 # TODO belongs in rlgraph
        
        for episode_idx in range(self.n_train_episodes):
            
            self.logger.info('Starting episode {}/{}'.format(episode_idx+1, self.n_train_episodes))

            # meta-data
            action_results = dict.fromkeys(['noop', 'duplicate_index', 'index'], 0)
            action_results_szs = []

            # queries + context
            if self.blackbox:
                episode_queries = queries
            else:
                episode_queries = queries[episode_idx * self.n_queries_per_episode:
                              (episode_idx + 1) * self.n_queries_per_episode]
            episode_queries = sort_queries(episode_queries)

            terminal_query_idx = len(episode_queries) - 1
            context = [] # list of list of attributes, corresponding to compound indices
            #index_set_size_prev = 0.0

            if self.dqn: self.task_graph.reset()
            self.system_environment.reset()
            
            for query_idx, query in enumerate(episode_queries):

                start = time.monotonic()

                if query_idx+1 % 5 == 0:
                    self.logger.info('Completed {}/{} queries'.format(query_idx+1,self.n_queries_per_episode))

                ##
                # get state
                ##
                acting_start = time.monotonic() 
                agent_state = self.converter.system_to_agent_state(query=query, 
                                                                   system_context=dict(indices=context))

                ##
                # get action for state
                ##
                agent_action = self.agent_api.get_action(agent_state)

                system_action = self.converter.agent_to_system_action(actions=agent_action, 
                                                                      meta_data=dict(query_cols=query.query_cols))
                acting_time = time.monotonic() - acting_start 
 
                ## 
                # take action (i.e. add index) and record reward (i.e. from query under updated index set)
                ##
                system_time, action_result = self.system_environment.act(dict(index=system_action, table=query.query_tbl))
                action_results[action_result] += 1
                action_results_szs.append(len(system_action))

                query_time, explained_idx = self.system_environment.execute(query)

                # record info about intxns - TODO refactor 
                existing_idx = self.get_intxn_opp(query, context) # check before updating context
                intxn_opp = 1 if existing_idx != [] else 0
                intxn_opp_taken = 1 if system_action == [] and explained_idx in [existing_idx[:i+1] for i in range(len(existing_idx))] else 0 # TODO explained_idx is sourced from EXPLAIN ANALYZE
                                                                                                                                              # and its cols (i.e. cols after Index Cond:) come from a subset of or complete index actually used
                                                                                                                                              # so existing_idx and explained_idx can be wrongfully equal if >1 index share those cols, though this is rare
                intersections.append([intxn_opp, intxn_opp_taken])
                self.logger.info('query cols: {} system action: {} explained_idx: {} existing_idx: {}'.format(list(query.query_cols), system_action, explained_idx, existing_idx))
                # e.g. intxn_opp_taken == 1: INFO / postgres_controller / query cols: ['L_DISCOUNT', 'L_EXTENDEDPRICE'] system action: [] explained_idx: ['L_EXTENDEDPRICE'] existing_idx: ['L_EXTENDEDPRICE', 'L_SHIPINSTRUCT']
    
                # reward a couple specific cases
                context_unaware = True if system_action != [] and explained_idx in [existing_idx[:i+1] for i in range(len(existing_idx))] else False
                column_unaware = True if system_action != [] and explained_idx not in [system_action[:i+1] for i in range(len(system_action))] else False

                index_set_size, _ = self.system_environment.system_status()
                
                ##
                # update agent
                ##
                updating_start = time.monotonic()
                agent_reward = self.converter.system_to_agent_reward(meta_data=dict(runtime=query_time,
                                                                                    index_size=index_set_size ,
                                                                                    adj = context_unaware or column_unaware))
                context.append(system_action)
                #index_set_size_prev = index_set_size
                next_agent_state = terminal = None # required because bootstrapping
                if self.dqn: 
                    terminal = query_idx == terminal_query_idx
                    next_query = episode_queries[query_idx + 1 if not terminal else query_idx]
                    next_agent_state = self.converter.system_to_agent_state(query=next_query, 
                                                                        system_context=dict(indices=context))

                loss = self.agent_api.observe(agent_state, agent_action, agent_reward, next_agent_state=next_agent_state, terminal=terminal)
                if loss is not None:
                    losses.append(loss)
                    if not self.dqn:
                        tensorboard.add_scalar('scalars/actor_loss', loss[0], step)
                        tensorboard.add_scalar('scalars/critic_loss', loss[1], step)
                    else:
                        tensorboard.add_scalar('scalars/loss', loss, step)
                    step += 1
                self.update_if_necessary()

                updating_time = time.monotonic() - updating_start

                # record results per step of episode
                times.append(time.monotonic() - start)
                agent_times.append(acting_time + updating_time)
                query_times.append(query_time)
                system_times.append(system_time)

                rewards.append(agent_reward)
                selectivities.append((self.get_selectivity(query.query_tbl, query.query_cols), # avg selectivity of query cols
                                      self.get_selectivity(query.query_tbl, system_action),    # avg selectivity of index cols in index suggested by agent
                                      self.get_selectivity(query.query_tbl, explained_idx)))   # avg selectivity of index cols in index actually used

            # record results per episode
            index_set_stats.append((action_results, np.mean(action_results_szs), index_set_size)) 

            # pretty print a summary
            self.logger.info("Completed episode: " \
                             "actions taken: " + ('{}:{:.2f} ' * len(tuple(action_results.items()))).format(*tuple(field for tup in action_results.items() for field in tup)) + \
                             "avg reward per step: {:.2f} ".format(np.mean(rewards[-self.n_queries_per_episode:])) + \
                             "avg query runtime: {:.2f} ".format(np.mean(query_times[-self.n_queries_per_episode:])) + \
                             "intxn opps: {}, opps taken: {} ".format(*tuple(np.sum(intersections[-self.n_queries_per_episode:], axis=0))) + \
                             "index set size {}/{:.0f}".format(len(self.system_environment.index_set), index_set_size))  
            tensorboard.add_scalar('scalars/reward', np.mean(rewards[-self.n_queries_per_episode:]), episode_idx)
            tensorboard.add_scalar('scalars/runtime', np.mean(query_times[-self.n_queries_per_episode:]), episode_idx)
            tensorboard.add_scalar('scalars/size', index_set_size, episode_idx)
            intxn_opps, intxn_opps_taken = np.sum(intersections[-self.n_queries_per_episode:], axis=0)
            if intxn_opps != 0:
                tensorboard.add_scalar('scalars/intxns', intxn_opps_taken / intxn_opps, episode_idx)

        # record results for all episodes 
        np.savetxt(os.path.join(result_dir, 'train_times.txt'), np.asarray(times), delimiter=',') # TODO replace np.savetxt with np.save
        np.savetxt(os.path.join(result_dir, 'train_agent_times.txt'), np.asarray(agent_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_system_times.txt'), np.asarray(system_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_query_times.txt'), np.asarray(query_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_rewards.txt'), np.asarray(rewards), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_selectivity.txt'), np.asarray(selectivities), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_intersections.txt'), np.asarray(intersections), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'losses.txt'), np.asarray(losses), delimiter=',')
        
        with open(os.path.join(result_dir, 'train_index_set_stats.csv'), 'a') as f:
            writer = csv.writer(f)
            for episode in index_set_stats:
                writer.writerow([*tuple(episode[0].values()), episode[1], episode[2]])
        
    def act(self, queries):
        """
        Emulate an episode on queries (i.e. test queries) without training to accumulate system actions

        Args:
            queries (list): SQLQuery objects 

        Returns:
            system actions (dict): records actions taken per query / query_idx
        """

        context = []
        idx_creation_time =[]

        system_actions = {}

        for query_idx, query in enumerate(queries):

            # start = time.monotonic()  

            agent_state = self.converter.system_to_agent_state(query=query, 
                                                               system_context=dict(indices=context))
            agent_action = self.agent_api.get_action(agent_state)
            
            system_action = self.converter.agent_to_system_action(actions=agent_action, 
                                                                  meta_data=dict(query_cols=query.query_cols))
                                                           
            idx_creation,_ = self.system_environment.act(dict(index=system_action, table=query.query_tbl))
            # end = time.monotonic()
            # TODO, Calculate the index creation time
            system_actions[query_idx] = system_action
            context.append(system_action)
            idx_creation_time.append(idx_creation)
        
        return idx_creation_time, system_actions

    def evaluate(self, queries, baseline=None, export=False):
        """
        Evaluates the execution of a set of queries (i.e. test queries).
        Assumes indices of interest are already set up. UPDATED.

        Args:
            queries (list): SQLQuery objects to execute and evaluate
            baseline (str): baseline identifier (e.g. default or full) 
        """

        runtimes = []
        index_set_sizes = []
        idx_creation_times = []
        eval_time = []

        start = time.monotonic()

        for episode_idx in range(self.n_test_episodes):
            
            episode_queries = queries[episode_idx * self.n_queries_per_episode:
                              (episode_idx + 1) * self.n_queries_per_episode]
            # episode_queries = sort_queries(episode_queries) # not in testing

            # install indices for agents
            if baseline is None:
                self.system_environment.reset()
                idx_creation_time, actions = self.act(episode_queries)
                idx_creation_times.append(idx_creation_time)

            for query in episode_queries:

                runtimes_per_query = []

                for _ in range(self.n_executions):
                    
                    query_time, _ = self.system_environment.execute(query)                
                    
                    runtimes_per_query.append(query_time)

                runtimes.append(runtimes_per_query)

            index_set_size, index_set = self.system_environment.system_status()
            index_set_sizes.append(index_set_size)
            
        eval_time.append(time.monotonic()-start)

        if export:
            tag = baseline if baseline is not None else 'dqn' if self.dqn else 'spg' # yikes
            result_dir = os.path.join(self.result_dir, tag)
            if not os.path.exists(result_dir): os.makedirs(result_dir)


            # idx creation time

            np.savetxt(os.path.join(result_dir, 'test_idx_times.txt'), np.asarray(idx_creation_times), delimiter=',')


            # runtimes
            np.savetxt(os.path.join(result_dir, 'test_query_times.txt'), np.asarray(runtimes), delimiter=',')
            
            # index set sizes
            np.savetxt(os.path.join(result_dir, 'test_index_set_sizes.txt'), np.asarray(index_set_sizes), delimiter=',')

            #Eval time
            np.savetxt(os.path.join(result_dir, 'eval_time.txt'), np.asarray(eval_time), delimiter=',')

            # TODO - rn saves stats for the final test episode
            
            with open(os.path.join(result_dir, 'test_index_set_stats.csv'), 'wb') as f:
                pickle.dump([index_set_size, index_set], f)
             
            # queries w/ any action taken 
            with open(os.path.join(result_dir, 'test_by_query.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                for query_idx, query in enumerate(episode_queries):
                    query_string, query_string_args = query.sample_query()
                    if baseline is None:
                        writer.writerow([query_string % query_string_args, actions.get(query_idx, ''), np.mean(runtimes[-self.n_queries_per_episode:][query_idx])])
                    else:
                        writer.writerow([query_string % query_string_args, np.mean(runtimes[-self.n_queries_per_episode:][query_idx])])

        return runtimes

    #
    # utils
    #
    def get_selectivity(self, tbl, cols):
        """Compute selectivity of set of cols, to compare query, agent's suggested index, system's index
        """
        return np.mean([self.system_environment.tbl_2_col_2_sel[tbl][col] for col in cols]) if list(cols) != [] else 0.0
    
    def get_intxn_opp(self, query, context):
        """Return index in context that query could use        
        
        a B-tree intersection opportunity occurs when any prefix of any permutation of columns is equal to any prefix of an existing index

        e.g. query with query.query_cols == [foo, bar] could use an index [bar, baz] (though probably the planner will only opt for this if 
        bar is sufficiently selective)
        """
        query_cols, n_query_cols = np.array(query.query_cols), len(query.query_cols)

        perm_idxs = itertools.permutations(range(n_query_cols))
        
        # suppose query.query_cols == [foo, bar, baz]
        # for each permutation of cols e.g. [baz, bar, foo]
        for perm_idx in perm_idxs:
            cols = list(query_cols[list(perm_idx)]) 

            # for each prefix of cols e.g. [baz, bar]
            for i in range(n_query_cols):
                cols_prefix = cols[:i+1]

                # check if shared by existing index e.g. [baz, bar, qux]
                for idx in context:
                    if cols_prefix == idx[:i+1]:
                        return idx

        return []

# TODO split out script
def main(argv):

    import gflags
    import json
    import numpy as np
    import torch

    #
    # logging
    # 
    import logging # import before rlgraph
    format_ = '%(levelname)s / %(module)s / %(message)s\n'
    formatter = logging.Formatter(format_)
    if logging.root.handlers == []:
        handler = logging.StreamHandler(stream=sys.stdout) # no different from basicConfig()
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logging.root.addHandler(handler)
    else:
    # handler has set up by default in rlgraph
        logging.root.handlers[0].setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.info('Starting controller script...')


    #
    # parsing - TODO command-line arg semantics aren't super clean
    #
    FLAGS = gflags.FLAGS
    
    # baselines rely on controller, which requires an agent, so set up dqn by default  
    gflags.DEFINE_boolean('dqn', True, 'dqn or spg, dqn by default for non-agent')
    gflags.DEFINE_boolean('blackbox', False, 'true to train and test on same workload (e.g. like an Atari agent')
    gflags.DEFINE_string('config', 'conf/dqn.json', 'config for agent, agent representations')
    gflags.DEFINE_string('experiment_config', '', 'config for experiment')
    gflags.DEFINE_boolean('generate_workload', True, 'set True to build train and test workloads')
    gflags.DEFINE_integer('seed', 0, 'seed for workload')
    gflags.DEFINE_string('result_dir', '../res/', 'base directory for workload and workload results')
    gflags.DEFINE_boolean('with_agent', True, 'set True for agent, False for non-agent baseline')
    gflags.DEFINE_boolean('default_baseline', True, 'set when with_agent is False')
    gflags.DEFINE_boolean('get_seed', False, '')
    # TODO should have separate paths for workloads, agents, results
    gflags.DEFINE_string('reevaluate_agent', '', 'path to agent if dqn, agent dir if spg')
    gflags.DEFINE_string('reevaluate_workload', '', 'path to workload to reevaluate agent on')
    
    
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    with open(FLAGS.config, 'r') as fh:
        config = json.load(fh)
    with open(FLAGS.experiment_config, 'r') as fh:
        experiment_config = json.load(fh)
    agent_config, schema_config = config['agent'], config['schema']
    logger.info('agent config: {}'.format(agent_config))
    logger.info('schema config: {}'.format(schema_config))
    logger.info('experiment config: {}'.format(experiment_config))

    result_dir = FLAGS.result_dir

    # build workload? or restore built workload?
    # if FLAGS.generate_workload:
    #     # if creating workload i.e. if starting a set of experiments, add a timestamp
    #     timestamp = time.strftime("%m-%d-%y_%H:%M", time.localtime())
    #     result_dir = os.path.join(result_dir, timestamp)
    #     if not os.path.exists(result_dir): os.makedirs(result_dir)

    
    # n.b. controller is required for agent-advised index or baseline index, b/c wraps system, workload
    controller = PostgresSystemController(
        dqn=FLAGS.dqn,
        blackbox=FLAGS.blackbox,
        agent_config=agent_config,
        schema_config=schema_config,
        experiment_config=experiment_config,
        result_dir=result_dir
    )

    # TODO crude way to suppress stdout from system_environment, this should be in a config somewhere 
    controller.system_environment.logger.setLevel(logging.WARNING) 


    seed = FLAGS.seed if FLAGS.seed != 0 else np.random.randint(1000)
    np.random.seed(seed)
    np.savetxt(os.path.join(result_dir, 'seed.txt'), np.asarray([seed]))

    logger.info('numpy seed: {}'.format(seed))
    logger.info('pytorch seed: {}'.format(torch.initial_seed())) # spg 

    # generate workload for first of a few experiments...
    if FLAGS.generate_workload:
        train, test = controller.generate_workload(export=True) 
    else:
        train, test = controller.restore_workload()
        # TODO! copy workload config?

    # ...which are one of
    
    # evaluate baseline 
    if not FLAGS.with_agent: 
        controller.reset_system() 
        if FLAGS.default_baseline:
            # 1º keys 
            controller.evaluate(test, baseline='default', export=True)
        else:
            # full - TODO hardcoded here 
            for tbl in experiment_config['tables']:
                for col in tpch_table_columns[tbl]:
                    controller.system_environment.act(dict(index=[col], table=tbl))
            controller.evaluate(test, baseline='full', export=True)
            
    # train agent, then evaluate agent's index
    elif FLAGS.with_agent:

        reeval = True if FLAGS.reevaluate_agent != '' else False
        if reeval:
            _, test = controller.restore_workload(path=FLAGS.reevaluate_workload)
            controller.agent_api.load_model(FLAGS.reevaluate_agent)
            controller.reset_system() 
            controller.evaluate(test, export=True) # TODO this replaces old results

        else:
            logger.info('TRAINING')
            controller.train(train)
            controller.system_environment.reset()

            logger.info('TESTING')
            controller.evaluate(test, export=True)
        
            controller.agent_api.save_model(result_dir + ('/dqn' if FLAGS.dqn else '/spg'))

    logger.info('RAN TO COMPLETION')



    if FLAGS.get_seed:

        # python3 src/common/postgres_controller.py --dqn=True --config=conf/dqn.json --experiment_config=conf/experiment.json --result_dir=.. --generate_workload=True --with_agent=False --default_baseline=True --get_seed=True &> ../seeds.log &

        seed_bound = 25

        controller.reset_system() 
        
        logger.info('scanning seeds -- with no indices')

        avg_wo_idxs = []
        upper_pct_wo_idxs = []
        for seed in range(seed_bound):
            np.random.seed(seed)
            if (seed+1) % 5 == 0: logger.info('completed {}/{} seeds'.format(seed+1,seed_bound))
            _, test = controller.generate_workload(export=False) 
            per_query_latencies = controller.evaluate(test, export=False)
            per_query_latency_μs = np.mean(per_query_latencies, axis=1) # (n_queries, n_executions_per_query)
            
            avg_wo_idxs.append(np.mean(per_query_latency_μs))
            upper_pct_wo_idxs.append(np.percentile(per_query_latency_μs, 90))

        for tbl in experiment_config['tables']:
            for col in tpch_table_columns[tbl]:
                controller.system_environment.act(dict(index=[col], table=tbl))

        logger.info('scanning seeds -- with indices')

        avg_w_idxs = []
        upper_pct_w_idxs = []
        for seed in range(seed_bound):
            np.random.seed(seed)
            if (seed+1) % 5 == 0: logger.info('completed {}/{} seeds'.format(seed+1,seed_bound))
            _, test = controller.generate_workload(export=False) 
            per_query_latencies = controller.evaluate(test, export=False)
            per_query_latency_μs = np.mean(per_query_latencies, axis=1)
            
            avg_w_idxs.append(np.mean(per_query_latency_μs))
            upper_pct_w_idxs.append(np.percentile(per_query_latency_μs, 90))

        workload_difficulty_wo_idxs = np.array(upper_pct_wo_idxs) / np.array(avg_wo_idxs)
        workload_difficulty_w_idxs = np.array(upper_pct_w_idxs) / np.array(avg_w_idxs)
        
        avg_speedup = np.array(avg_wo_idxs) / np.array(avg_w_idxs)
        upper_pct_speedup = np.array(upper_pct_wo_idxs) / np.array(upper_pct_w_idxs)

        seeds = np.arange(seed_bound)

        res = np.concatenate((np.expand_dims(seeds, axis=1), 
                              np.expand_dims(workload_difficulty_wo_idxs, axis=1),
                              np.expand_dims(workload_difficulty_w_idxs, axis=1),
                              np.expand_dims(avg_speedup, axis=1),
                              np.expand_dims(upper_pct_speedup, axis=1)), axis=1)
        with open('seeds.txt', 'w') as f:
            np.savetxt(f, res, fmt='%.2f')

if __name__ == "__main__":
    main(sys.argv)
