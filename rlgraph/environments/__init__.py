# Copyright 2018/2019 The RLgraph authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rlgraph.environments.environment import Environment
from rlgraph.environments.deterministic_env import DeterministicEnv
from rlgraph.environments.gaussian_density_as_reward_env import GaussianDensityAsRewardEnv
from rlgraph.environments.grid_world import GridWorld
from rlgraph.environments.openai_gym import OpenAIGymEnv
from rlgraph.environments.random_env import RandomEnv
from rlgraph.environments.vector_env import VectorEnv
from rlgraph.environments.sequential_vector_env import SequentialVectorEnv

Environment.__lookup_classes__ = dict(
    deterministic=DeterministicEnv,
    deterministicenv=DeterministicEnv,
    gaussiandensity=GaussianDensityAsRewardEnv,
    gaussiandensityasreward=GaussianDensityAsRewardEnv,
    gaussiandensityasrewardenv=GaussianDensityAsRewardEnv,
    gridworld=GridWorld,
    gridworldenv=GridWorld,
    openai=OpenAIGymEnv,
    openaigym=OpenAIGymEnv,
    openaigymenv=OpenAIGymEnv,
    random=RandomEnv,
    randomenv=RandomEnv,
    sequentialvector=SequentialVectorEnv,
    sequentialvectorenv=SequentialVectorEnv
)

try:
    import deepmind_lab

    # If import works: Can import our Adapter.
    from rlgraph.environments.deepmind_lab import DeepmindLabEnv

    Environment.__lookup_classes__.update(dict(
        deepmindlab=DeepmindLabEnv,
        deepmindlabenv=DeepmindLabEnv,
    ))
    # TODO travis error on this, investigate.
except Exception:
    pass


try:
    import mlagents

    # If import works: Can import our Adapter.
    from rlgraph.environments.mlagents_env import MLAgentsEnv

    Environment.__lookup_classes__.update(dict(
        mlagents=MLAgentsEnv,
        mlagentsenv=MLAgentsEnv,
    ))
    # TODO travis error on this, investigate.
except Exception:
    pass


__all__ = ["Environment"] + \
          list(set(map(lambda x: x.__name__, Environment.__lookup_classes__.values())))
