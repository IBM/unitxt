from typing import List, Optional, Union

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
