from typing import List

from .artifact import Artifact
from .collections import Collection
from .dataclass import OptionalField
from .loaders import Loader
from .operator import StreamingOperator
from .splitters import RandomSampler, Sampler
from .task import Task


class TaskCard(Artifact):
    """TaskCard delineates the phases in transforming the source dataset into a model-input, and specifies the metrics for evaluation of model-output.

    Attributes:
        loader: specifies the source address and the loading operator that can access that source and transform it into a unitxt multistream.

        preprocess_steps: list of unitxt operators to process the data source into a model-input.

        task: specifies the fields (of the already (pre)processed instance) making the inputs, the fields making the outputs, and the metrics to be used for evaluating the model output.

        templates: format strings to be applied on the input fields (specified by the task) and the output fields. The template also carries the instructions and the list of postprocessing steps, to be applied to the model output.
    """

    loader: Loader
    preprocess_steps: List[StreamingOperator] = None
    task: Task
    templates: Collection = None
    sampler: Sampler = OptionalField(default_factory=RandomSampler)
