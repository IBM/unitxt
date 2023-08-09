from typing import List, Optional, Union

from .artifact import Artifact
from .collections import Collection
from .loaders import Loader
from .operator import StreamingOperator
from .task import FormTask
from .splitters import Sampler, RandomSampler
from .dataclass import OptionalField

class TaskCard(Artifact):
    loader: Loader
    task: FormTask
    preprocess_steps: Optional[List[Union[StreamingOperator, str]]] = None
    templates: Collection = None
    instructions: Collection = None
    sampler: Sampler = OptionalField(default_factory=RandomSampler)


class ICLCard(Artifact):
    demos_pool_name: str = "demos_pool"
    demos_pool_size: int = None
    demos_field: str = "demos"
    num_demos: int = None
    sampler_type: str = "random"
    instruction_item: Union[str, int] = None
    template_item: Union[str, int] = None
