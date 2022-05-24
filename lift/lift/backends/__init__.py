
from lift.backends.backend_util import set_learning_rate
from lift.backends.backend_util import convert_to_backend
from lift import BACKEND

# Import and map any agents here if desired.
if BACKEND == "tensorforce":
    from tensorforce.agents import DQFDAgent
    AGENTS = dict(
        dqfd=DQFDAgent,
        dqfd_agent=DQFDAgent
    )
elif BACKEND == "rlgraph":
    from rlgraph.agents import DQFDAgent
    AGENTS = dict(
        dqfd=DQFDAgent,
        dqfd_agent=DQFDAgent
    )