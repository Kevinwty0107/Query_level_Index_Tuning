import rlgraph
from spg_agent import SPGAgent

# dqn
from rlgraph.spaces.int_box import IntBox
from rlgraph.spaces.containers import Dict
from rlgraph.agents import Agent
import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../../lift')) 
print(sys.path)
from lift.controller.system_controller import SystemController

from lift.rl_model.task import Task # update
from lift.rl_model.task_graph import TaskGraph



vocab_size = 6
state_size = 6
steps =100


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

state_spec = IntBox(low=0, high=vocab_size, shape=(state_size,))
n_outputs = 1+3
action_spec = {}
for i in range(3):
    action_spec['candidate_index_column{}'.format(i)] = IntBox(low=0, high=n_outputs)
action_spec = Dict(action_spec, add_batch_rank=True)
print()
task_graph = TaskGraph()
task = Task(agent_config, state_space=state_spec, action_space=action_spec)


