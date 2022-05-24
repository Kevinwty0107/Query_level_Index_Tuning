a = [12,4]
print(a.copy())
import rlgraph
#from rlgraph.spaces.int_box import IntBox
#from rlgraph.spaces.containers import Dict
import os
import gflags
import sys
import json
import pickle
head, tail = os.path.split(__file__)
with open('/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_Code/RLAutoIndex/conf/dqn.json', 'r') as f:
    config = json.load(f)
agent_config = config['agent']
print(head,'\n', tail)
sys.path.insert(0, os.path.join(head, '../')) # /src
print(sys.path) # src

with open(os.path.join('/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_Code/res/01-29-22_00:56/dqn', 'test_index_set_stats.csv'), 'rb') as f:
    test_index_set_stats = pickle.load(f)
        
index_set_size = test_index_set_stats[0]
print(index_set_size )

print(dict(abc=[1,2,3],edf=[24]))