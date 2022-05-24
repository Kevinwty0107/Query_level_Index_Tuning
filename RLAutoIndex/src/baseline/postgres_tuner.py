#!/usr/local/bin/python 
# OpenTuner requires 2 not 3

import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../')) # /src for access to src/common

from common.postgres_data_source import PostgresDataSource
from common.postgres_system_environment import PostgresSystemEnvironment

from common.tpch_util import tpch_table_columns

import argparse 
import csv
import logging
import numpy as np
import pdb
import pickle
import time 

import opentuner
from opentuner.api import TuningRunManager
from opentuner.measurement.interface import DefaultMeasurementInterface
from opentuner.search.manipulator import ConfigurationManipulator, IntegerParameter

class PostgresTuner():
    """
    An OpenTuner interface for Postgres indexing

    OpenTuner API usage:
    - define a parameter search space for parameters to tune. 
    - define a run loop, in which:
        - OpenTuner returns a configuration, a particular assignment of parameters in search space
        - we execute and evaluate that configuration on the system
        - and return the reward to OpenTuner, refining the search  
    
    Search space:

    1st: 
        - parameter per action taken for candidate index column (allow 3 index columns per index) per index (allow index per query)
        - size... this is (up to) 3 candidate index columns per query * 100 episodes * 20 queries per episodes = 6000
        
    2nd:
        - parameter per action taken for candidate index column per index / indexing decision in an episode
        - size? 3 * 20
        - idea?
            - analogous to how RL agent works. exposed to 20 queries per episode.
            - and evaluate on a random subset of set of train queries 


    Sourced from: 
        - lift/case_studies/{mongodb,mysql}/ 
        - https://github.com/jansel/opentuner/blob/master/examples/py_api/api_example.py
    
    """
    def __init__(self, train_queries, test_queries, experiment_config, result_dir, system_environment):
        """
        Args:
            train_queries (list of queries (SQLQuery queries))
            test_queries (list of queries (SQLQuery queries))
            
            result_dir (str) 
            system (PostgresSystemEnvironment): encapsulates environment
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info('Setting up OpenTuner...')

        self.train_queries = train_queries
        self.test_queries = test_queries
        self.experiment_config = experiment_config

        self.max_size = experiment_config['max_size']
        self.size_weight = experiment_config['size_weight']
        self.max_runtime = experiment_config['max_runtime']
        self.runtime_weight = experiment_config['runtime_weight']
        self.reward_penalty = experiment_config['reward_penalty']

        self.system_environment = system_environment

        self.result_dir = result_dir


        # 2nd search space
        self.n_idxs = experiment_config['n_queries_per_episode']
        self.n_cols_per_idx = 4 # TODO hardcode
        
        self.tbls = experiment_config['tables']
        self.n_idxs_per_tbl = int(self.n_idxs / len(self.tbls)) 

        # maps tbls to mapping of tbl's column indices to tbl's columns, per search space representation
        self.tbl_2_col_idx_2_col = {}
        for tbl in self.tbls:
            self.tbl_2_col_idx_2_col[tbl] = {}
            for col_idx, col in enumerate(tpch_table_columns[tbl].keys()):
                self.tbl_2_col_idx_2_col[tbl][col_idx + 1] = col # + 1 b/c 0 is noop
        
        # apictio
        sys.argv = [sys.argv[0]] # opentuner expects own args
        parser = argparse.ArgumentParser(parents=opentuner.argparsers())
        args = parser.parse_args()
        manipulator = self.build_search_space()
        interface = DefaultMeasurementInterface(args=args,
                                                manipulator=manipulator)
        self.api = TuningRunManager(interface, args)

    def build_search_space(self):
        """
        Set up search space via ConfigurationManipulator, which is responsible for choosing configurations across runs,
        where a configuration is a particular assignment of the parameters. 

        Tightly coupled with agent/system representations used in src.
        
        Repeating from above:
        *initial* search space 
        - parameter per action (candidate index column in index, where candidate index column is one of the query columns) 
          per query
        - see commit 8f94c4b

        *updated* search space
        - parameter per action (same, but here candidate index column is not one of query columns but any of columns in query table)
          per allowed index / indexing decision

         so, at risk of redundancy, an "action" refers to the particular assignment of a parameter, which is 
         an integer indicating noop or a column for a candidate index column 
         
        """

        self.logger.info('Building OpenTuner search space...')      

        self.idx_2_idx_cols = {} # store candidate index columns (OpenTuner operates on this) with candidate index (system operates on this)
        self.idx_2_tbl = {}

        manipulator = ConfigurationManipulator()

        for tbl in self.tbls:
            n_cols_per_tbl = len(tpch_table_columns[tbl])

            for candidate_idx in range(self.n_idxs_per_tbl):        
                idx_id = "tbl_{}_idx_{}".format(tbl, candidate_idx)
                idx_col_ids = []

                for candidate_idx_col in range(self.n_cols_per_idx):
                    idx_col_id = idx_id + "_idx_col_{}".format(candidate_idx_col)
                    idx_col_ids.append(idx_col_id)

                    manipulator.add_parameter(IntegerParameter(idx_col_id, 0, n_cols_per_tbl))
            
                self.idx_2_idx_cols[idx_id] = idx_col_ids 
                self.idx_2_tbl[idx_id] = tbl 
        
        self.logger.info("... actions are: {}".format(self.idx_2_idx_cols))
        return manipulator

    def run(self, n_iterations):
        """
        Runs OpenTuner search

        run loop - https://github.com/jansel/opentuner/blob/master/examples/py_api/api_example.py
        "desired result" vs "result" - http://groups.csail.mit.edu/commit/papers/2014/ansel-pact14-opentuner.pdf
        """
        self.logger.info('Running OpenTuner...')     
        idx_creation_times =[]
        # search
        for i in range(n_iterations):
            self.logger.info('iteration {}/{}...'.format(i+1, n_iterations))
            start = time.time()
            
            desired_result = self.api.get_next_desired_result()
            configuration = desired_result.configuration.data            
            
            idx_time,reward, _ = self.act(configuration)
            result = opentuner.resultsdb.models.Result(time=-1*reward)
            self.api.report_result(desired_result, result)

            self.logger.info('...received reward {} after {} seconds'.format(reward, time.time()-start))
            idx_creation_times.append(idx_time)

        # runtimes
        np.savetxt(os.path.join(self.result_dir, 'test_idx_times.txt'), np.asarray(idx_creation_times), delimiter=',')
        # best from search
        best = self.api.get_best_configuration()
        self.eval_best(best)
        self.system_environment.reset()


    def act(self, cfg):
        """
        Get reward for current configuration i.e. recommended index.

        Args:
            cfg: current configuration returned by OpenTuner search, maps parameter to parameter value
        """
        self.system_environment.reset()
        context = []
        episode_reward = 0
        idx_creation_time =[]

        # for each candidate index, extract decisions for candidate index columns
        for idx_id, idx_col_ids in self.idx_2_idx_cols.items():

            # get integer (i.e. integer value of IntegerParameter)
            start = time.monotonic()  

            actions = []
            for idx_col_id in idx_col_ids:
                action = cfg[idx_col_id]
                actions.append(action)

            # get corresponding attribute for integer
            system_action = []
            tbl = self.idx_2_tbl[idx_id]
            for action in actions:
                if action != 0: # noop
                    system_action.append(self.tbl_2_col_idx_2_col[tbl][action])

            #
            # execute
            #
            self.system_environment.act(dict(index=system_action, table=tbl))
            context.append(system_action)
            end = time.monotonic()

            idx_creation_time.append(end-start)

        #
        # evaluate on a randomly selected subset of workload
        # 
        train_query_sample = np.random.choice(self.train_queries, 
                                              size=max(int(len(self.train_queries) / 10), 1)) # TODO add to config
        for query in train_query_sample:
            query_time, _ = self.system_environment.execute(query, explain=True) # TODO
            index_set_size, index_set = self.system_environment.system_status()

            reward = self.system_to_agent_reward(data=dict(runtime=query_time,
                                                           index_size=index_set_size))

            episode_reward += reward

        # N.B. context can't be used(!) 
        return idx_creation_time, episode_reward, context

    def eval_best(self, config):
        self.logger.info('Evaluating best OpenTuner configuration...')
        # create configuration
        idx_creation_time, reward, context = self.act(config)

        runs = self.experiment_config['n_executions']

        runtimes = []
        for query in self.test_queries:
            runtimes_per_query = []
            for _ in range(runs):
                query_time, _ = self.system_environment.execute(query, explain=True)
            
                runtimes_per_query.append(query_time)
            runtimes.append(runtimes_per_query)

        index_set_size, index_set = self.system_environment.system_status()

        # runtimes
        np.savetxt(os.path.join(self.result_dir, 'test_query_times.txt'), np.asarray(runtimes), delimiter=',')
        
        # index set, index set size 
        with open(os.path.join(self.result_dir, 'test_index_set_stats.csv'), 'wb') as f:
            pickle.dump([index_set_size, index_set], f)

    #
    # Auxiliary
    #
    def system_to_agent_reward(self, data):
        """
        Same as in Converter, but didn't want to have Schema / Converter here
        """
        runtime, index_size = data["runtime"], data["index_size"]
        reward = - (self.runtime_weight * runtime) - (self.size_weight * index_size)

        if runtime > self.max_runtime:
            reward -= self.reward_penalty
        if index_size > self.max_size:
            reward -= self.reward_penalty
        return reward


# run temporarily as a script from here while debugging, e.g. python src/baseline/postgres_tuner.py --config=conf/dqn.json --data_dir=../res/
def main(argv):

    import gflags
    import json

    # gather args
    FLAGS = gflags.FLAGS
    gflags.DEFINE_string('experiment_config', '', 'experiment_config')
    gflags.DEFINE_string('data_dir', '../res/', 'train, test queries in directory; results to be stored in subdirectory of directory ')

    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    with open(FLAGS.experiment_config, 'r') as fh:
        experiment_config = json.load(fh)
    
    # build training interfaces, retrieve data from training
    system_environment = PostgresSystemEnvironment(tbls=experiment_config['tables'])

    # suppress info system actions (there are len(train_queries) * 3 of them...)
    system_logger = logging.getLogger('postgres_system_environment')
    system_logger.setLevel(logging.WARNING)
    
    workload_spec = { "tables": experiment_config['tables'],
                      "scale_factor": 1,
                      "n_selections": experiment_config['n_selections']}
    data_source = PostgresDataSource(workload_spec=workload_spec)

    train_queries = data_source.import_data(FLAGS.data_dir, label='train')
    test_queries = data_source.import_data(FLAGS.data_dir, label='test')

    result_dir = os.path.join(FLAGS.data_dir, 'tuner')
    if not os.path.exists(result_dir): os.makedirs(result_dir)    

    tuner = PostgresTuner(
        train_queries=train_queries,
        test_queries=test_queries,
        experiment_config=experiment_config,
        result_dir=result_dir,
        system_environment=system_environment,
    )

    tuner.run(experiment_config['n_opentuner_search_steps'])

if __name__ == "__main__":
    main(sys.argv)
    
