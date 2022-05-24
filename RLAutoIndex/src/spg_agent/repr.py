# spg
from spg_agent import SPGAgent

# dqn
from rlgraph.spaces.int_box import IntBox
from rlgraph.spaces.containers import Dict
from rlgraph.agents import Agent
import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../../lift')) 
from lift.controller.system_controller import SystemController

from lift.rl_model.task import Task # update
from lift.rl_model.task_graph import TaskGraph


from collections import deque
import itertools
import numpy as np
import pdb
import time
import torch 


#
# vocabulary
#
pad_token = 0
noop_token = 1
foo_token = 2
bar_token = 3 
baz_token = 4
foo_idx_token = 5
bar_idx_token = 6
baz_idx_token = 7

#
# agent representations
# 
class RepresentationBuilder():
    """
        Utility that:
        - accepts user-specified queries and actions for those queries (that should be rewarded). 
            - this requires specifying the vocab of tokens, and specifying the queries / actions for queries in terms of those vocab tokens
        - augments representations (i.e. builds out numpy.ndarray for DQN, torch.Tensor for SPG), returns a function to sample queries, and returns a function to reward actions given the sampled query

        Representations are same as in Schemas / Converters for DQN and SPG agents, of course
    """

    def build_dqn(self, queries, actions_per_query, K=12, prob=0.5):
        """
    
        Args:
            queries (list of lists of ints): queries
            actions_per_query (list of list of ints): actions for corresponding queries to be rewarded
            K (int): size of state / query representation
            prob (float): probability with which to sample from queries (vs. random query)
        """

        # sampler
        def get_query():

            # sample a specified queries
            if np.random.rand() < prob:
                query_idx = np.random.randint(0,len(queries))
                query = queries[query_idx]
                
            # or sample a non-specified / random query
            else:
                query_idx = len(queries) 
                query = np.random.choice([foo_token,bar_token,baz_token], size=3, replace=False) # TODO build realistic non-reward queries
                query = query.tolist()

            # build out 
            pad = K - len(query)
            if pad > 0: 
                query = query + [pad_token] * pad    
            else: 
                query = query[:K]
        
            query = np.asarray(query)
            return query_idx, query

        # reward i.e. get reward if action taken is action specified for query, per query_idx 
        def get_reward(query_idx, action):

            if (query_idx == len(queries) and action == []) \
            or (query_idx != len(queries) and actions_per_query[query_idx] == action):  # won't be out of range b/c of short-circuit 
                reward = 1
            else:
                reward = -1

            return reward
    
        return get_query, get_reward

    def build_spg(self, queries, actions_per_query, N=4, K=12, prob=0.5):
        """
        Args:
            as above
            N, K as in SPG
        """

        def get_query():

            assert noop_token == 1, pad_token == 0
            query_template = np.zeros(shape=(N,K), dtype=int) # N,K representation, non-attribute cols should be padding by default
            query_template[:,0] = noop_token # attribute cols should be noops by default

            # sample a specified queries
            if np.random.rand() < prob:
                query_idx = np.random.randint(0,len(queries))
            # or sample a non-specified / random query
                query = queries[query_idx]
            else:
                query_idx = len(queries)
                query = np.random.choice([foo_token,bar_token,baz_token], size=3, replace=False) # TODO build realistic non-reward queries    

            # build out i.e. superimpose on N,K tensor
            query = np.array(query)
            if len(query.shape) == 1: query = query.reshape(-1,1)
            query_template[:query.shape[0], :query.shape[1]] = query
            query = query_template
            
            query = torch.tensor(query).float().view(1,N,K)
            return query_idx, query

        def get_reward(query_idx, action):

            if (query_idx == len(queries) and action == []) \
            or (query_idx != len(queries) and actions_per_query[query_idx] == action):
                reward = 1
            else:
                reward = -1

            return reward
    
        return get_query, get_reward

#
# spg
#
def run_spg(exp, steps=25000):

    N = 4
    K = 2

    #
    # queries, rewards for actions per query
    # 
    _, spg_queries, actions = data(exp)
    repr_builder = RepresentationBuilder()
    get_query, get_reward = repr_builder.build_spg(spg_queries, actions, K=K, prob=0.67)

    #
    # agent
    #

    import json
    with open('/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_Code/RLAutoIndex/conf/spg.json', 'r') as f:
        config = json.load(f)
    agent_config = config['agent']

    # adjustments?
    agent_config['N'] = N
    agent_config['K'] = K
    agent_config['embed_dim'] = 32
    agent_config['rnn_dim'] = 32
    agent_config['bidirectional'] = True
    agent_config['epsilon_decay_steps'] = steps

    agent = SPGAgent(agent_config)

    print("params: {}".format( sum(param.numel() for param in agent.actor.parameters() if param.requires_grad) +\
                               sum(param.numel() for param in agent.critic.parameters() if param.requires_grad) ))

    #
    # train agent
    #

    step = 0
    record = []
    running_avg_reward = deque(maxlen=1000)
    start = time.time()
    
    while step < steps:
        step += 1

        if step != 0 and step % 1000 == 0:
            print('running avg reward after {}/{} steps is {}'.format(step, steps, np.mean(running_avg_reward)))
            record.append((step, np.mean(running_avg_reward), time.time() - start))

        query_idx, query = get_query()

        action = agent.get_action(query)
        agent_action = action['agent_action'].tolist()

        reward = get_reward(query_idx, agent_action)        
        running_avg_reward.append(reward)

        agent.observe(query, action, reward)
    
    return record

#
# dqn
#
def run_dqn(exp, steps=25000, combinatorial=False):
    
    #
    # can't account for all configurations, but be sure agent is of a reasonably small size 
    # 

    vocab_size = 6
    state_size = 6

    #
    # queries, rewards for actions per query
    # 
    dqn_queries, _, actions = data(exp)
    repr_builder = RepresentationBuilder()
    get_query, get_reward = repr_builder.build_dqn(dqn_queries, actions, K=state_size, prob=0.67)

    
    #
    # agent
    #
    import json # config is a bit big to copy
    with open('/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_Code/RLAutoIndex/conf/dqn.json', 'r') as f:
        config = json.load(f)
    agent_config = config['agent']

    # any further adjustments?
    agent_config['memory_spec']['type']='replay' 
    agent_config['exploration_spec']['epsilon_spec']['decay_spec']['num_timesteps'] = int(steps * .75)

    agent_config['network_spec'][0]['embed_dim'] = 32 # reduce capacity
    agent_config['network_spec'][2]['units'] = 32
    agent_config['network_spec'][0]['vocab_size'] = vocab_size


    # replicate representations defined in Schema
    state_spec = IntBox(low=0, high=vocab_size, shape=(state_size,))

    if not combinatorial:
        n_outputs = 1+3
        action_spec = {}
        for i in range(3):
            action_spec['candidate_index_column{}'.format(i)] = IntBox(low=0, high=n_outputs)
        action_spec = Dict(action_spec, add_batch_rank=True)
    else:
        perm_idx_2_perm = []
        for r in range(3+1): 
            perm_idx_2_perm.extend(itertools.permutations(range(3),r=r))
        perm_idx_2_perm = list(map(list, perm_idx_2_perm)) # [[], [1], [2], [3], [1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

        # action is a scalar corresponding to a particular permutation of query attributes
        action_spec = IntBox(low=0, high=len(perm_idx_2_perm))


    task_graph = TaskGraph()
    task = Task(agent_config, state_space=state_spec, action_space=action_spec)
    task_graph.add_task(task)
    task_graph.get_task("").unwrap().timesteps = 0
    with open('/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_Code/RLAutoIndex/conf/experiment.json', 'r') as fh:
        experiment_config = json.load(fh)
    controller = SystemController(agent_config,experiment_config ) # have to have for updates...
    controller.task_graph = task_graph
    controller.set_update_schedule(agent_config["update_spec"])

    print("params: {}".format(task.agent.graph_builder.num_trainable_parameters)) # TODO yikes

    #
    # train agent
    #
    step = 0; steps = steps 
    record = []
    running_avg_reward = deque(maxlen=1000)
    start = time.time()
    while step < steps:
        step += 1

        if step != 0 and step % 1000 == 0:
            print('running avg reward after {}/{} steps is {}'.format(step, steps, np.mean(running_avg_reward)))
            record.append((step, np.mean(running_avg_reward), time.time() - start))

        query_idx, query = get_query()
        
        agent_action = task_graph.act_task("", query, apply_preprocessing=True)
        
        # replicate representation conversions defined in Converter
        # hack - same as how query_cols are stored with query in actual training loop
        attr_tokens = [foo_token, bar_token, baz_token]
        n_attrs = len([attr_token for attr_token in query[:3] if attr_token in attr_tokens]) # count tokens that are column tokens
        
        if not combinatorial:
            action = []
            for key in ['candidate_index_column{}'.format(i) for i in range(3)]:
                action_val = agent_action[key][0]
                if action_val != 0: # if is not noop
                    if n_attrs > action_val - 1: # if is a valid action
                        col = query[:n_attrs][action_val - 1]
                        if col not in action:
                            action.append(col)

        else:
            action = []
            perm_idx = agent_action 
            perm = perm_idx_2_perm[perm_idx]
            
            if len(perm) == n_attrs: # ignore case like query==[foo], permutation of query==[1,2]
                for query_attr_idx in perm:
                    if n_attrs > query_attr_idx: # ignore case like query==[foo], permutation of query==[1] b/c there is only 0th attribute, not 1st attribute
                        col = query[:n_attrs][query_attr_idx]
                        # if col not in action: # no repeats in this representation
                        action.append(col)

        reward = get_reward(query_idx, action)        
        running_avg_reward.append(reward)

        # TODO what to do with s_t+1???
        task_graph.observe_task("", query, agent_action, [], reward, query, False)
        controller.update_if_necessary()

    return record


def data(exp):
    """
        DEFINE QUERIES, ACTIONS FOR QUERIES HERE
    """

    #
    # map single query to index, should have K small
    #
    if exp == 1:
     
        dqn_queries = [[foo_token, bar_token, baz_token]] # i.e. this should be augmented to a np.ndarray of shape (K,), not to be confused with K in context of SPG 
        spg_queries = [[foo_token, bar_token, baz_token, noop_token]] # i.e. this should be augmented to a torch.Tensor of shape (4,K)
        actions = [[foo_token, bar_token]]

    #
    # map single query to index, include token indicating op, should have K slightly bigger
    #
    # TODO

    #
    # map several queries to indices, again including token indicating op TODO
    # 
    elif exp == 2:

        dqn_queries = [[foo_token],
                    [bar_token],
                    [foo_token, baz_token]]
        spg_queries = [[foo_token, 
                        noop_token, 
                        noop_token, 
                        noop_token],
                    [bar_token, 
                        noop_token, 
                        noop_token, 
                        noop_token],
                    [foo_token, 
                        baz_token, 
                        noop_token, 
                        noop_token]]
        actions = [[foo_token], [bar_token], [foo_token, baz_token]]

    #
    # map single query to index, if index has not been constructed 
    # 
    elif exp == 3:
            
        dqn_queries = [[foo_token],
                    [foo_token, foo_idx_token]]

        spg_queries = [[[foo_token, pad_token], # foo with no foo index
                        [noop_token, pad_token], 
                        [noop_token, pad_token], 
                        [noop_token, pad_token]], 
                        
                    [[foo_token, foo_idx_token], # foo with foo index
                        [noop_token, pad_token], 
                        [noop_token, pad_token], 
                        [noop_token, pad_token]]]
        actions = [[foo_token], []]

    return dqn_queries, spg_queries, actions


if __name__ == "__main__":

    # for i in $(seq 1 3); do python3 repr.py --dqn=True --exp=$i --steps=50000; python3 repr.py --dqn=False --exp=$i --steps=50000; done

    """
    import gflags, sys
    FLAGS = gflags.FLAGS
    gflags.DEFINE_boolean('dqn', True, 'DQN if True, SPG if False')
    gflags.DEFINE_integer('exp', 1, 'experiment')
    gflags.DEFINE_integer('steps', 25000, 'steps')
    argv = FLAGS(sys.argv)
    run_dqn(FLAGS.exp, steps=FLAGS.steps, combinatorial=False) if FLAGS.dqn else run_spg(FLAGS.exp, steps=FLAGS.steps) 
    """
    
    run_dqn(1, steps=5000, combinatorial=False)
    #run_spg(1, steps=5000)
"""    
    for exp in [1,2,3]:
        
        runs = []
        for i in range(5):
            print("#### EXP {}/{}, DQN COMB, RUN {}/{} ####".format(exp, 3, i+1, 5))
            runs.append(run_dqn(exp, steps=100000, combinatorial=True))
        np.save('repr-exp-{}-dqn-comb.npy'.format(exp), runs)

        runs = []
        for i in range(5):
            print("#### EXP {}/{}, DQN NON-COMB, RUN {}/{} ####".format(exp, 3, i+1, 5))
            runs.append(run_dqn(exp, steps=100000, combinatorial=False))
        np.save('repr-exp-{}-dqn.npy'.format(exp), runs)

        runs = []
        for i in range(5):
            print("#### EXP {}/{}, SPG, RUN {}/{} ####".format(exp, 3, i+1, 5))
            runs.append(run_spg(exp, steps=100000))
        np.save('repr-exp-{}-spg.npy'.format(exp), runs)
"""