from lift.case_studies.mongodb.deprecated.execution.executor import Executor
from lift.case_studies.mongodb.deprecated.execution.random_executor import RandomExecutor
from lift.case_studies.mongodb.deprecated.execution.pretrain_executor import PretrainExecutor
from lift.case_studies.mongodb.deprecated.execution.online_executor import OnlineExecutor
from lift.case_studies.mongodb.deprecated.execution.noop_executor import NoOpExecutor

executors = dict(
    random=RandomExecutor,
    online=OnlineExecutor,
    pretrain=PretrainExecutor,
    noop=NoOpExecutor
)

__all__ = [
    'Executor',
    'RandomExecutor',
    'PretrainExecutor',
    'OnlineExecutor',
    'NoOpExecutor'
]
