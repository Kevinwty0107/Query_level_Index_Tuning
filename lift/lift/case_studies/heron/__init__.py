from lift.case_studies.heron.heron_simple_model_generator import \
    HeronSimpleModelGenerator
from lift.case_studies.heron.heron_system_schema import HeronSystemSchema
from lift.case_studies.heron.heron_simple_agent_schema import \
    HeronSimpleAgentSchema
from lift.case_studies.heron.reward_generator import LinearTLRewardGenerator, \
    ResourceUsageRewardGenerator
from lift.case_studies.heron.load_gen import AlternatingLoadGenerator, \
    AlternatingFixedLoadGenerator, UpDownLoadGenerator, DownUpLoadGenerator
from lift.case_studies.heron.interval_generator import \
    ConstantIntervalGenerator, UniformIntervalGenerator
from lift.case_studies.heron.heron_agent_model_generator import \
    HeronAgentModelGenerator
from lift.case_studies.heron.heron_agent_schema import HeronAgentSchema
from lift.case_studies.heron.aperiodic_load_process import AperiodicLoadProcess
from lift.case_studies.heron.periodic_load_process import PeriodicLoadProcess
from lift.case_studies.heron.heron_rule_based_model_generator import HeronRuleBasedModelGenerator
from lift.case_studies.heron.heron_agent_full_state_schema import \
    HeronFullStateSchema
from lift.case_studies.heron.heron_agent_full_state_model_generator import \
    HeronFullStateModelGenerator
from lift.case_studies.heron.heron_full_sequence_model_generator import \
    HeronFullSequenceModelGenerator

heron_schemas = dict(
    system=HeronSystemSchema,
    simple=HeronSimpleAgentSchema,
    agent=HeronAgentSchema,
    full_state=HeronFullStateSchema
)

heron_reward_generators = dict(
    linear=LinearTLRewardGenerator,
    resource=ResourceUsageRewardGenerator
)

heron_model_generators = dict(
    simple=HeronSimpleModelGenerator,
    rules=HeronRuleBasedModelGenerator,
    agent=HeronAgentModelGenerator,
    full_state=HeronFullStateModelGenerator,
    full_sequence=HeronFullSequenceModelGenerator
)

# The combined load generator requires this dictionary to be defined.
# Thus we define it as here, THEN import the combined load generator and 
# add it to the dictionary. This means that combined load generators cannot
# be stacked. 
heron_load_generators = dict(
    alternating=AlternatingLoadGenerator,
    alternating_fixed=AlternatingFixedLoadGenerator,
    updown=UpDownLoadGenerator,
    downup=DownUpLoadGenerator,
)
# import the combined load generator
from lift.case_studies.heron.combined_load_gen import CombinedLoadGenerator

heron_load_generators['combined'] = CombinedLoadGenerator

heron_interval_generators = dict(
    constant=ConstantIntervalGenerator,
    uniform=UniformIntervalGenerator
)
heron_load_processes = dict(
    periodic=PeriodicLoadProcess,
    aperiodic=AperiodicLoadProcess
)
