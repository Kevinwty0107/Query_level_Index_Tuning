import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../..'))
from lift.lift.rl_model.converter import Converter
from lift.lift.rl_model.state import State
from sklearn import preprocessing
import numpy as np
import torch

class PostgresConverter(Converter):
    """
        converts agent, system representations of actions, rewards, states
        
        this is specific to an SPG agent
    """
    
    def __init__(self, experiment_config, schema):

        self.system_spec = schema.get_system_spec()
        
        # unpack
        self.N, self.K = schema.N, schema.K # TODO get rid of this
        self.vocab = self.system_spec['vocab']
        self.pad_token = self.system_spec['pad_token']
        self.noop_token = self.system_spec['noop_token']
        self.idx_token = self.system_spec['idx_token']


        # reward
        self.max_size = experiment_config['max_size']
        self.size_weight = experiment_config['size_weight']
        self.max_runtime = experiment_config['max_runtime']
        self.runtime_weight = experiment_config['runtime_weight']
        self.reward_penalty = experiment_config['reward_penalty']
        

    def system_to_agent_action(self):
        pass 

    def agent_to_system_action(self, actions, meta_data=None):
        """
        assume that the agent has applied P, and we receive by our choosing / convention 
        the tokens / integer tokens for attributes in the result. 
        
        simply retrieve the tokens for the non-noop query attributes  
        """
        actions = actions['agent_action'] # TODO replace

        # ah also have to get token from token integer index
        vocab = {v: k for k, v in self.vocab.items()}
        return [vocab[query_attr] for query_attr in actions if vocab[query_attr] != self.noop_token]

    def system_to_agent_reward(self, meta_data):
        """
        Reward as a weighted aggregation of (index) space and (query under index) time
        """

        runtime = meta_data["runtime"]
        size = meta_data["index_size"]
        adj = meta_data["adj"]
        reward = - (self.runtime_weight * runtime) - (self.size_weight * size)
        
        # TODO experimental
        if adj:
            reward *= 1.25

        if runtime > self.max_runtime:
            reward -= self.reward_penalty

        if size > self.max_size:
            reward -= self.reward_penalty

        return reward


    def system_to_agent_state(self, query, system_context):
        """
        Build N by K representation for SPG 

        recall representation where N = n query attributes allowed + 1

        suppose 
            query.tokens() == [foo, =, baz, =]
            system_context == [[baz], [foo, bar], [foo]]

            vocab
                pad_token => 0
                noop_token => 1
                idx_token => 2
                foo => 3
                bar => 4
                baz => 5 
                foo_idx => 6
                bar_idx => 7
                baz_idx => 8
                = => 9
                
                tensor representation we want is:
                [[3 9 2 7 6 2 3 0... up to K tokens],
                 [5 9 2 8 0... ],
                 [1 0 ...],
                 [1 0 ...]]
                
            TODO how to represent the table?
        """

        # TODO
        # should there be a noop or a pad in row j where there's no query attribute in row j-1?
        # I'd think a noop rather than a pad, so that noop and pad aren't conflated "semantically"
        assert self.vocab[self.pad_token] == 0
        assert self.vocab[self.noop_token] == 1
        state = np.concatenate((np.ones(shape=(self.N, 1)),
                                np.zeros(shape=(self.N, self.K-1))), axis=1)


        query_tokens = query.as_tokens().copy()


        for query_col_idx, query_col in enumerate(query.query_cols):
            tokens = []
            
            # 1. get query state for this query attribute
            # hacky, but col and col ops only exposed together as tokens like 
            # ['O_ORDERDATE', '<', 'O_CUSTKEY', '=']
            col = query_tokens[2*query_col_idx]
            op_for_col = query_tokens[2*query_col_idx+1]
            tokens.append(col)
            tokens.append(op_for_col)
            
            # 2. get context state (relevant context state) for this query attribute
            for index in system_context['indices']:
                if [col for col in index if col == query_col] != []:
                    tokens.append(self.idx_token) # demarcate / delimit
                    tokens.extend(list(map(lambda col: col + '_idx', index)))

            indexed_tokens = [self.vocab[token] for token in tokens]
            
            if len(indexed_tokens) > self.K:
                indexed_tokens = indexed_tokens[:self.K]
            
            state[query_col_idx, :len(indexed_tokens)] = indexed_tokens 

        # torchify
        state = torch.tensor(state).float().view(1,self.N, self.K)

        # TODO any reason to use this State abstraction? whoops yes the type
        return state # State(value=state, meta_data=dict(query_columns=query.query_cols))

        