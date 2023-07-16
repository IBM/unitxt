from typing import List, Optional, Union, Dict

from .artifact import Artifact
from .instructions import InstructionsDict, InstructionsList
from .loaders import Loader
from .operator import StreamingOperator
from .task import FormTask
from .templates import TemplatesDict, TemplatesList


class TaskCard(Artifact):
    loader: Loader
    task: FormTask
    preprocess_steps: Optional[List[StreamingOperator]] = None
    templates: Union[TemplatesList, TemplatesDict] = None
    instructions: Union[InstructionsList, InstructionsDict] = None

class ICLCard(Artifact):
    demos_pool_name: str = "demos_pool"
    demos_pool_size: int = None
    demos_field: str = "demos"
    num_demos: int = None
    sampler_type: str = "random"
    instruction_item: Union[str, int] = None
    template_item: Union[str, int] = None

