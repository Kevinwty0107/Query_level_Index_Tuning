
import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.rl_model.schema import Schema
sys.path.insert(0, os.path.join(head, '..')) # src
from common.tpch_util import tpch_table_columns

class PostgresSchema(Schema):

    def __init__(self, schema_config):
        """
        Specific to SPG agent
        Input / output representations for DQN agent were specified in schema + converter
        Representations for SPG agent are specified in schema somewhat + converter  
        """
        
        self.schema_config = schema_config # recall, this is consumed only by this object
        self.N = schema_config['N'] # save globally here?
        self.K = schema_config['K']
        self.tbls = schema_config['tables']
        self.cols = [col for tbl in self.tbls for col in tpch_table_columns[tbl].keys()]

        self.system_spec = {}
        #self.states_spec = {} # n.b. was required for rlgraph representation 
        #self.actions_spec = {} n.b. was required for rlgraph representation

        self.query_ops = ["SELECT", "FROM", "WHERE"]
        self.query_selection_ops = ["AND", "LIKE", "IN", ">", "=", "<"]

        self.build_input_tokens()
        # self.build_output_tokens()

    def build_input_tokens(self):
        """
        Tokenizes vocab used in system state representations, so as to subsequently embed state in SPG

        """
        vocab = {}
        vocab_size = 0

        #
        # special tokens
        #
        pad_token = 'pad' 
        vocab[pad_token] = vocab_size
        vocab_size += 1
        
        noop_token = 'noop'
        vocab[noop_token] = vocab_size
        vocab_size += 1
        
        idx_token = 'idx' # delimits / demarcates compound indices
        vocab[idx_token] = vocab_size
        vocab_size += 1

        #
        # state tokens
        #
        
        # state = ...
        # ... query 
        for col in self.cols:
            vocab[col] = vocab_size
            vocab_size += 1
        
        for op in self.query_selection_ops:
            vocab[op] = vocab_size
            vocab_size += 1

        # ... + context
        for col in self.cols:
            vocab[col + '_idx'] = vocab_size
            vocab_size += 1

        self.system_spec['vocab'] = vocab
        self.system_spec['vocab_size'] = len(vocab)
        self.system_spec['pad_token'] = pad_token
        self.system_spec['noop_token'] = noop_token
        self.system_spec['idx_token'] = idx_token

    def build_output_tokens(self):
        pass

    def get_actions_spec(self):
        pass

    def get_states_spec(self):
        pass

    def get_system_spec(self):
        return self.system_spec