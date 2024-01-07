from typing import List

from .artifact import Artifact
from .collections import Collection
from .dataclass import OptionalField
from .loaders import Loader
from .operator import StreamingOperator
from .splitters import RandomSampler, Sampler
from .task import FormTask


class TaskCard(Artifact):
    loader: Loader
    task: FormTask
    preprocess_steps: List[StreamingOperator] = None
    templates: Collection = None
    instructions: Collection = None
    sampler: Sampler = OptionalField(default_factory=RandomSampler)
