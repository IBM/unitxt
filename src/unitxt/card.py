from typing import Dict, List, Union

from .artifact import Artifact
from .dataclass import OptionalField
from .loaders import Loader
from .operator import StreamingOperator
from .splitters import RandomSampler, Sampler
from .task import Task
from .templates import Template, TemplatesDict, TemplatesList


class TaskCard(Artifact):
    """TaskCard delineates the phases in transforming the source dataset into model input, and specifies the metrics for evaluation of model output.

    Args:
        loader:
            specifies the source address and the loading operator that can access that source and transform it into a unitxt multistream.
        preprocess_steps:
            list of unitxt operators to process the data source into model input.
        task:
            specifies the fields (of the already (pre)processed instance) making the inputs, the fields making the outputs, and the metrics to be used for evaluating the model output.
        templates:
            format strings to be applied on the input fields (specified by the task) and the output fields. The template also carries the instructions and the list of postprocessing steps, to be applied to the model output.
        default_template:
            a default template for tasks with very specific task dataset specific template
    """

    loader: Loader
    preprocess_steps: List[StreamingOperator] = None
    task: Task
    templates: Union[
        TemplatesDict, TemplatesList, Dict[str, Template], List[Template]
    ] = None
    default_template: Template = None
    sampler: Sampler = OptionalField(default_factory=RandomSampler)
