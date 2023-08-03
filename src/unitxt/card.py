from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .artifact import Artifact
from .collections import Collection
from .instructions import InstructionsDict, InstructionsList
from .loaders import Loader
from .normalizers import NormalizeListFields
from .operator import StreamingOperator
from .operators import AddFields, MapInstanceValues
from .task import FormTask
from .templates import TemplatesDict, TemplatesList


class TaskCard(Artifact):
    loader: Loader
    task: FormTask
    preprocess_steps: Optional[List[Union[StreamingOperator, str]]] = None
    templates: Collection = None
    instructions: Collection = None


class ICLCard(Artifact):
    demos_pool_name: str = "demos_pool"
    demos_pool_size: int = None
    demos_field: str = "demos"
    num_demos: int = None
    sampler_type: str = "random"
    instruction_item: Union[str, int] = None
    template_item: Union[str, int] = None
