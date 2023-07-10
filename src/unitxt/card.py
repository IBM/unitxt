from .artifact import Artifact
from .operator import StreamingOperator
from .templates import TemplatesList, TemplatesDict
from .instructions import InstructionsList, InstructionsDict
from .loaders import Loader
from .task import FormTask

from typing import Union, List, Optional


class TaskCard(Artifact):
    loader: Loader
    task: FormTask
    preprocess_steps: Optional[List[StreamingOperator]] = None
    templates: Union[TemplatesList, TemplatesDict] = None
    instructions: Union[InstructionsList, InstructionsDict] = None
