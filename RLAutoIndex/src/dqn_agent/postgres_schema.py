import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.rl_model.schema import Schema
sys.path.insert(0, os.path.join(head, '..')) # src
from common.tpch_util import tpch_table_columns

from rlgraph.spaces import IntBox, Dict

class PostgresSchema(Schema):

    def __init__(self, schema_config):
        """
        Schema builds schemas for different data structures for Q-network, like 
        states_spec (spec for state representation), and actions_spec (spec for action representation)

        Boilerplated from lift/case_studies/{mongodb,mysql}/ 

        Args:
            schema_config (dict): spec for schema; see controller for creation
        """

        self.schema_config = schema_config
        self.tbls = schema_config['tables']
        self.cols = [col for tbl in self.tbls for col in tpch_table_columns[tbl].keys()]

        self.system_spec = {}
        self.states_spec = {}
        self.actions_spec = {}

        # currently, all queries have same structure
        self.include_default_operators = schema_config.get("include_default", False)
        self.query_ops = ["SELECT", "FROM", "WHERE"]
        self.query_selection_ops = ["AND", "LIKE", "IN", ">", "=", "<"]

        # TODO update vars / var names?
        self.input_sequence_size = schema_config["input_sequence_size"]
        self.max_fields_per_index = schema_config.get("max_fields_per_index", 3)
        self.build_input_tokens()
        self.build_output_tokens()


    def build_input_tokens(self):
        """
        Tokenizes vocabulary used for state representations for input to Q-network by 
        assigning integers to vocab words (query operators, query operands 
        i.e. attributes represented in workload)

        Exposed through self.system_spec and self.states_spec
        """
        
        self.system_spec["state_dim"] = self.input_sequence_size
        vocab = {}
        vocab_size = 0
        
        #
        # tokenize
        #

        # special tokens
        pad_token = 'pad'
        vocab[pad_token] = vocab_size
        vocab_size += 1

        ## state = ...
        ## ... query
        # operands
        for col in self.cols:
            vocab[col] = vocab_size
            vocab_size += 1
        
        # operators
        if self.include_default_operators:
            for op in self.query_ops:
                vocab[op] = vocab_size
                vocab_size += 1
        
        for op in self.query_selection_ops:
            vocab[op] = vocab_size
            vocab_size += 1

        ## ... + context TODO
        for col in self.cols:
            vocab[col + '_idx'] = vocab_size
            vocab_size += 1

        # delimits / demarcates compound indices 
        idx_token = 'idx'
        vocab[idx_token] = vocab_size

        #
        # specific input schema, i.e. a vector of vocabulary tokens in Z^n, n=specified input size, to be embedded in embedding layer
        #
        self.states_spec = IntBox(
                            low=0,
                            high=vocab_size,
                            shape=(self.input_sequence_size,)
                          )
        self.system_spec['vocab'] = vocab
        self.system_spec['vocab_size'] = len(vocab)
        self.system_spec['index_token'] = idx_token
        self.system_spec['pad_token'] = pad_token
        
    def build_output_tokens(self):
        """
        Tokenizes vocabulary used for action representations for output of Q-network

        Exposed through self.system_spec and self.actions_spec

        Recall action representation maps index field (a candidate index field) to a decision
        e.g. 
            suppose allow indices on up to 3 cols, allow indices to be ASC or DESC
            then the action is specified in [0,6], where 0 corresponds to noop, 
            {1,2} correspond to an ASC or DESC index on 1st query attribute, 
            {3,4} correspond to an ASC or DESC index on 2nd query attribute, etc.

            {0:1, 1:0, 2:0} is an action specifying an index (ascending index) on 1st query attributes,
            and noops for the 2 remaining allowed columns for the compound index 

        n.b. actions_spec comes from action branching architectures https://arxiv.org/abs/1711.08946
        TODO dig deeper into that

        """

        noop_idx = 0
        idxs = []
        self.actions_spec = {}

        # not sure whether ASC / DESC can be specified
        # see LIFT paper for this representation in particular
        n_outputs = 1 + self.max_fields_per_index # 1 + 2 * self.max_fields_per_index
        for i in range(self.max_fields_per_index):
            idxs.append('index_column{}'.format(i))
            self.actions_spec['index_column{}'.format(i)] = IntBox(low=0, high=n_outputs)
        
        # ?
        self.actions_spec = Dict(self.actions_spec, add_batch_rank=True)

        self.system_spec['idxs'] = idxs
        self.system_spec['n_outputs'] = n_outputs
        self.system_spec['noop_idx'] = noop_idx
        self.system_spec['max_fields_per_index'] = self.max_fields_per_index

    def get_actions_spec(self):
        return self.actions_spec

    def get_states_spec(self):
        return self.states_spec

    def get_system_spec(self):
        return self.system_spec
