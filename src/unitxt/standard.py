import itertools
import json
import sys
from typing import Any, Dict, Generator, List, Optional, Union

from .artifact import fetch_artifact
from .augmentors import Augmentor, NullAugmentor
from .card import TaskCard
from .collections_operators import GetLength
from .dataclass import Field, InternalField, NonPositionalField, OptionalField
from .deprecation_utils import deprecation
from .error_utils import UnitxtError
from .formats import Format, SystemFormat
from .generator_utils import ReusableGenerator
from .logging_utils import get_logger
from .operator import (
    MultiStreamOperator,
    SequentialOperator,
    SourceSequentialOperator,
    StreamingOperator,
)
from .operators import Set, StreamRefiner
from .schema import FinalizeDataset
from .serializers import SingleTypeSerializer
from .settings_utils import get_constants, get_settings
from .splitters import ConstantSizeSample, RandomSizeSample, Sampler
from .stream import MultiStream
from .system_prompts import EmptySystemPrompt, SystemPrompt
from .task import Task
from .templates import (
    ApplyRandomTemplate,
    ApplySingleTemplate,
    Template,
    TemplatesList,
)
from .type_utils import isoftype
from .utils import LRUCache, recursive_copy

constants = get_constants()
settings = get_settings()
logger = get_logger()


# Used to give meaningful name to recipe steps
class CreateDemosPool(MultiStreamOperator):
    from_stream: str = None
    demos_pool_size: int = None
    demos_removed_from_data: bool = None
    to_field: str = constants.demos_pool_field

    # flake8: noqa: B007
    def process(self, multi_stream: MultiStream) -> MultiStream:
        # generate the demos_pool as a selection of demos_pool_size distinct instances
        # (distinct by their "input_fields" field). The selection is taken from stream named from_stream.
        # The selected instances are later treated as ordinary instances or not, depending on parameter
        # demos_removed_from_data.
        # The selection of instances is done from the first instances of the stream named from_stream.
        # instances that are not distinct from previously selected demo instances, are kept aside, to be later
        # treated like all the remaining instances of stream from_stream.
        if self.from_stream not in multi_stream:
            raise ValueError(
                f"Input multi-stream is missing a stream named '{self.from_stream}' to take demo instances from for the demos_pool."
            )
        if (
            self.demos_removed_from_data is not None
            and self.demos_removed_from_data is True
            and (self.demos_pool_size == sys.maxsize)
        ):
            # going to consume the whole of input stream named self.from_stream for demo instances,
            # and not let demos instances to behave as regular instances. so self.from_stream
            # ends here its life as an input stream that is expected to reach the end of the recipe
            if len(multi_stream) == 1:
                raise ValueError(
                    f"The single input stream, '{self.from_stream}' is to be wholly consumed for generating demos, and no instance is left to use these demos."
                )
        from_stream = multi_stream[self.from_stream]
        demos_pool = []
        input_fields_of_demos_pool = []
        not_selected_from_from_stream = []
        for num_scanned, instance in enumerate(from_stream):
            if "input_fields" not in instance:
                raise ValueError(f"'input_fields' field is missing from '{instance}'.")
            try:
                input_fields_signature = json.dumps(
                    instance["input_fields"], sort_keys=True
                )
            except TypeError:
                input_fields_signature = str(instance["input_fields"])
            if input_fields_signature in input_fields_of_demos_pool:
                not_selected_from_from_stream.append(instance)
                continue
            demos_pool.append(instance)
            input_fields_of_demos_pool.append(input_fields_signature)
            if len(demos_pool) >= self.demos_pool_size:
                break

            # for backward compatibility, do not throw exception here if demos pool is smaller than expected.
            # Delay that for the event (if occurs) that Sample is not be able to sample num_demos demos.

        # to avoid endless recursion in case of not demos_removed_from_data
        demos_pool = recursive_copy(demos_pool)

        set_demos_pool = Set(fields={self.to_field: demos_pool})
        if (
            self.demos_removed_from_data is not None
            and self.demos_removed_from_data is False
        ):
            # all input instances go out. No one is "killed" because selected as demo
            return set_demos_pool(multi_stream)

        if (
            self.demos_removed_from_data is not None
            and self.demos_removed_from_data is True
        ):
            if self.demos_pool_size == sys.maxsize:
                # consume the whole of input stream self.from_stream, just for demos, and do not
                # take any of its instances to behave as a non-demo instance, i.e., a regular instance
                # that consume the demos
                out_ms = MultiStream(
                    {
                        stream_name: multi_stream[stream_name]
                        for stream_name in multi_stream
                        if stream_name != self.from_stream
                    }
                )
                return set_demos_pool(out_ms)

        #  self.demos_removed_from_data and not consume the whole of self.from_stream just for demos
        def from_stream_generator(
            first_layer: list, ms: MultiStream, stream_name: str, start: int
        ) -> Generator:
            yield from first_layer
            yield from itertools.islice(ms[stream_name], start, None)

        new_streams = {}
        for stream_name in multi_stream:
            if stream_name == self.from_stream:
                new_streams[stream_name] = ReusableGenerator(
                    generator=from_stream_generator,
                    gen_kwargs={
                        "first_layer": not_selected_from_from_stream,
                        "ms": multi_stream,
                        "stream_name": self.from_stream,
                        "start": num_scanned + 1,
                    },
                )
            else:
                new_streams[stream_name] = ReusableGenerator(
                    generator=from_stream_generator,
                    gen_kwargs={
                        "first_layer": [],
                        "ms": multi_stream,
                        "stream_name": stream_name,
                        "start": 0,
                    },
                )

        ms = MultiStream.from_generators(new_streams)
        return set_demos_pool(ms)


class AddDemosPool(MultiStreamOperator):
    demos_pool: List[Dict[str, Any]]
    demos_pool_field_name: str = constants.demos_pool_field

    def process(self, multi_stream: MultiStream) -> MultiStream:
        set_demos_pool = Set(fields={self.demos_pool_field_name: self.demos_pool})
        return set_demos_pool(multi_stream)


class DatasetRecipe(SourceSequentialOperator):
    """This class represents a standard recipe for data processing and preparation.

    This class can be used to prepare a recipe.
    with all necessary steps, refiners and renderers included. It allows to set various
    parameters and steps in a sequential manner for preparing the recipe.

    Args:
        card (TaskCard):
            TaskCard object associated with the recipe.
        template (Template, optional):
            Template object to be used for the recipe.
        system_prompt (SystemPrompt, optional):
            SystemPrompt object to be used for the recipe.
        loader_limit (int, optional):
            Specifies the maximum number of instances per stream to be returned from the loader (used to reduce loading time in large datasets)
        format (SystemFormat, optional):
            SystemFormat object to be used for the recipe.
        metrics (List[str]):
            list of catalog metrics to use with this recipe.
        postprocessors (List[str]):
            list of catalog processors to apply at post processing. (Not recommended to use from here)
        group_by (List[Union[str, List[str]]]):
            list of task_data or metadata keys to group global scores by.
        train_refiner (StreamRefiner, optional):
            Train refiner to be used in the recipe.
        max_train_instances (int, optional):
            Maximum training instances for the refiner.
        validation_refiner (StreamRefiner, optional):
            Validation refiner to be used in the recipe.
        max_validation_instances (int, optional):
            Maximum validation instances for the refiner.
        test_refiner (StreamRefiner, optional):
            Test refiner to be used in the recipe.
        max_test_instances (int, optional):
            Maximum test instances for the refiner.
        demos_pool_size (int, optional):
            Size of the demos pool. -1 for taking the whole of stream 'demos_taken_from'.
        demos_pool(List[Dict[str, Any]], optional):
            a list of instances to make the demos_pool
        num_demos (int, optional):
            Number of demos to add to each instance, to become part of the source to be generated for this instance.
        demos_taken_from (str, optional):
            Specifies the stream from where the demos are taken. Default is "train".
        demos_field (str, optional):
            Field name for demos. Default is "demos".
            The num_demos demos selected for an instance are stored in this field of that instance.
        demos_pool_field_name (str, optional):
            field name to maintain the demos_pool, until sampled from, in order to make the demos.
            Defaults to constants.demos_pool_field.
        demos_removed_from_data (bool, optional):
            whether to remove the demos taken to demos_pool from the source data, Default is True
        sampler (Sampler, optional):
            The Sampler used to select the demonstrations when num_demos > 0.
        skip_demoed_instances (bool, optional):
            whether to skip pushing demos to an instance whose demos_field is
            already populated. Defaults to False.
        steps (List[StreamingOperator], optional):
            List of StreamingOperator objects to be used in the recipe.
        augmentor (Augmentor) :
            Augmentor to be used to pseudo randomly augment the source text
        instruction_card_index (int, optional):
            Index of instruction card to be used for preparing the recipe.
        template_card_index (int, optional):
            Index of template card to be used for preparing the recipe.

    Methods:
        prepare():
            This overridden method is used for preparing the recipe
            by arranging all the steps, refiners, and renderers in a sequential manner.

    Raises:
        AssertionError:
            If both template and template_card_index are specified at the same time.
    """

    # Base parameters
    card: TaskCard = None
    task: Task = None
    template: Union[Template, List[Template], TemplatesList] = None
    system_prompt: SystemPrompt = Field(default_factory=EmptySystemPrompt)
    format: Format = None
    serializer: Union[SingleTypeSerializer, List[SingleTypeSerializer]] = None

    # Additional parameters
    template_card_index: int = NonPositionalField(default=None)
    metrics: List[str] = NonPositionalField(default=None)
    postprocessors: List[str] = NonPositionalField(default=None)

    group_by: List[Union[str, List[str]]] = []

    loader_limit: int = None

    max_train_instances: int = None
    max_validation_instances: int = None
    max_test_instances: int = None

    train_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)
    validation_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)
    test_refiner: StreamRefiner = OptionalField(default_factory=StreamRefiner)

    demos_pool_size: int = None
    demos_pool: List[Dict[str, Any]] = None
    num_demos: Optional[Union[int, List[int]]] = 0
    demos_removed_from_data: bool = True
    demos_pool_field_name: str = constants.demos_pool_field

    demos_taken_from: str = "train"
    demos_field: str = constants.demos_field
    sampler: Sampler = None

    # do not push demos to instances whose "demos" field is already populated
    skip_demoed_instances: bool = False

    augmentor: Union[Augmentor, List[Augmentor]] = OptionalField(default=None)

    steps: List[StreamingOperator] = InternalField(default_factory=list)

    # shared class cache
    _demos_pool_cache = LRUCache(max_size=10)

    def before_process_multi_stream(self):
        super().before_process_multi_stream()

    @property
    def max_demos_size(self):
        if isinstance(self.num_demos, list):
            return max(self.num_demos)
        return self.num_demos

    def verify(self):
        super().verify()

        if self.task is None and self.card is None:
            raise ValueError("Set card or task in the recipe")

        if self.card is None and (
            self.num_demos > 0 or self.demos_pool_size is not None
        ):
            raise ValueError(
                "To use num_demos and demos_pool_size in recipe set a card."
            )

        if self.use_demos:
            if self.demos_pool_size is None or self.demos_pool_size < 1:
                raise ValueError(
                    "When using demonstrations both num_demos and demos_pool_size should be assigned with positive integers."
                )
            if self.demos_pool_size < self.max_demos_size + 1:
                raise ValueError(
                    f"num_demos (got: {self.max_demos_size}) should not exceed demos_pool_size - 1 (got: {self.demos_pool_size}), (-1: to always allow filtering of a demo identical to the processed instance)."
                )
            if (
                (not self.demos_pool)
                and (self.demos_pool_size != sys.maxsize)
                and self.loader_limit
                and (self.demos_pool_size > self.loader_limit)
            ):
                raise ValueError(
                    f"demos_pool_size should not exceed loader_limit ({self.loader_limit}), Got demos_pool_size={self.demos_pool_size}"
                )

        if self.loader_limit:
            if self.max_test_instances and self.max_test_instances > self.loader_limit:
                raise ValueError(
                    f"max_test_instances should not exceed loader_limit ({self.loader_limit}), Got max_test_instances={self.max_test_instances}"
                )
            if (
                self.max_validation_instances
                and self.max_validation_instances > self.loader_limit
            ):
                raise ValueError(
                    f"max_validation_instances should not exceed loader_limit ({self.loader_limit}), Got max_validation_instances={self.max_validation_instances}"
                )
            if (
                self.max_train_instances
                and self.max_train_instances > self.loader_limit
            ):
                raise ValueError(
                    f"max_train_instances should not exceed loader_limit ({self.loader_limit}), Got max_train_instances={self.max_train_instances}"
                )
        if self.metrics is not None and not isinstance(self.metrics, List):
            raise ValueError(
                f"metrics must be a list of metrics.  Got metrics = {self.metrics}"
            )
        if self.postprocessors is not None and not isinstance(
            self.postprocessors, List
        ):
            raise ValueError(
                f"post processors must be a list of post processor.  Got postprocessors = {self.postprocessors}"
            )

        if self.format is not None and not isinstance(self.format, Format):
            raise ValueError(
                f"format parameter must be a list of of class derived from Format.  Got format = {self.format}"
            )
        if self.template is None:
            raise ValueError(
                "You must set in the recipe either `template`, `template_card_index`."
            )

        if isinstance(self.template, list):
            for template in self.template:
                self.verify_template(template)
        else:
            self.verify_template(self.template)

        if self.serializer is not None:
            if not isinstance(self.serializer, list):
                self.serializer = [self.serializer]
            self.template.serializer.add_serializers(self.serializer)

    def prepare_refiners(self):
        self.train_refiner.max_instances = self.max_train_instances
        self.train_refiner.apply_to_streams = ["train"]
        self.processing.steps.append(self.train_refiner)

        self.validation_refiner.max_instances = self.max_validation_instances
        self.validation_refiner.apply_to_streams = ["validation"]
        self.processing.steps.append(self.validation_refiner)

        self.test_refiner.max_instances = self.max_test_instances
        self.test_refiner.apply_to_streams = ["test"]
        self.processing.steps.append(self.test_refiner)

    def verify_template(self, template):
        if not isinstance(template, Template):
            raise ValueError(
                f"template argument must be an object of type Template. Got template = {template}"
            )

    def set_pipelines(self):
        self.loading = SequentialOperator(
            __description__="Loading the data from the data source."
        )
        self.metadata = SequentialOperator(
            __description__="Adding metadata (e.g. format, system prompt, template)  "
        )
        self.standardization = SequentialOperator(
            __description__="Standardizing the raw dataset fields to task field definition."
        )

        self.processing = SequentialOperator(
            __description__="Setting task fields (and selecting demos per sample if needed)."
        )
        self.verbalization = SequentialOperator()
        self.verbalization.__description__ = "Verbalizing the input to the model and gold references to the 'source', 'target' and 'references' fields."
        self.finalize = SequentialOperator()
        self.finalize.__description__ = "Adding post processors. Removing intermediate fields. Creating the final output dataset."

        self.steps = [
            self.loading,
            self.metadata,
            self.standardization,
            self.processing,
            self.verbalization,
            self.finalize,
        ]

        self.inference_instance = SequentialOperator()

        self.inference_instance.steps = [
            self.metadata,
            self.processing,
        ]

        self.inference_demos = SourceSequentialOperator()

        self.inference_demos.steps = [
            self.loading,
            self.metadata,
            self.standardization,
        ]

        self.inference = SequentialOperator()

        self.inference.steps = [self.processing, self.verbalization, self.finalize]

    def production_preprocess(self, task_instances):
        ms = MultiStream.from_iterables({constants.inference_stream: task_instances})
        return list(self.metadata(ms)[constants.inference_stream])

    @property
    def has_custom_demos_pool(self):
        return self.demos_pool_size is not None and (
            self.demos_pool_size > 0 or self.demos_pool_size == -1
        )

    @property
    def use_demos(self):
        return self.num_demos is not None and self.max_demos_size > 0

    def produce(self, task_instances):
        """Use the recipe in production to produce model ready query from standard task instance."""
        self.before_process_multi_stream()

        ms = MultiStream.from_iterables({constants.inference_stream: task_instances})
        # does not hurt to set metadata
        # task_instances are assumed to be as if passed through self.standardization
        ms = self.metadata(ms)
        if not self.use_demos:
            # go with task_instances all the way, it does not need other streams:
            ms = self.inference(ms)
            return list(ms[constants.inference_stream])

        streams = self.inference_demos()
        # streams stopped before processing
        # ms is ready to join, it will get the demos from streams
        streams[constants.inference_stream] = ms[constants.inference_stream]
        # multi_stream = MultiStream(streams)
        multi_stream = self.inference(streams)
        return list(multi_stream[constants.inference_stream])

    def reset(self):
        self.reset_pipeline()

    def reset_pipeline(self):
        if self.format is None:
            if settings.default_format is not None:
                self.format, _ = fetch_artifact(settings.default_format)
            else:
                self.format = SystemFormat()

        if self.card and self.card.preprocess_steps is None:
            self.card.preprocess_steps = []

        if self.task is None:
            self.task = self.card.task

        self.set_pipelines()

        if self.card is not None:
            loader = self.card.loader
            if self.loader_limit:
                loader.loader_limit = self.loader_limit
                # logger.info(f"Loader line limit was set to  {self.loader_limit}")
            self.loading.steps.append(loader)

            # This is required in case loader_limit is not enforced by the loader
            if self.loader_limit:
                self.loading.steps.append(
                    StreamRefiner(max_instances=self.loader_limit)
                )

        self.metadata.steps.append(
            Set(
                fields={
                    "recipe_metadata/system_prompt": self.system_prompt,
                    "recipe_metadata/format": self.format,
                }
            )
        )

        if self.card:
            self.standardization.steps.extend(self.card.preprocess_steps)

        self.processing.steps.append(self.task)

        if self.augmentor is not None and not isoftype(self.augmentor, NullAugmentor):
            if (
                self.card.task.augmentable_inputs is None
                or len(self.task.augmentable_inputs) == 0
            ):
                raise UnitxtError(
                    f"You specified augmentor in the recipe but the got task without augmentable_inputs: {self.task}"
                )

            if not isinstance(self.augmentor, list):
                self.augmentor = [self.augmentor]

            for augmentor in self.augmentor:
                augmentor.set_fields(self.card.task.augmentable_inputs)
                self.processing.steps.append(augmentor)

        # for backward compatibility, consume the demos instances even if not pushed into demos field of the ordinary instances,
        # in order to use the very same ordinary instances as in back releases.
        # one example of consume but not used, and indeed skips over a problematic (json-wise) input:
        # prepare/cards/rag/end_to_end/clapnq.py
        if self.has_custom_demos_pool:
            if self.demos_pool:
                self.processing.steps.append(
                    AddDemosPool(
                        demos_pool=self.demos_pool,
                        demos_pool_field_name=self.demos_pool_field_name,
                    )
                )
            else:
                self.processing.steps.append(
                    CreateDemosPool(
                        from_stream=self.demos_taken_from,
                        demos_pool_size=self.demos_pool_size
                        if self.demos_pool is None
                        else None,
                        demos_removed_from_data=self.demos_removed_from_data,
                        to_field=self.demos_pool_field_name,
                    )
                )

        if self.use_demos:
            if self.sampler is None:
                if self.card.sampler is None:
                    raise ValueError(
                        "Unexpected None value for card.sampler. "
                        "To use num_demos > 0, please set a sampler on the TaskCard."
                    )
                self.sampler = self.card.sampler

        self.prepare_refiners()

        if self.use_demos:
            if isinstance(self.num_demos, int):
                self.verbalization.steps.append(
                    ConstantSizeSample(
                        from_field=self.demos_pool_field_name,
                        to_field=self.demos_field,
                        sampler=self.sampler,
                        sample_size=self.num_demos,
                        skip_demoed_instances=self.skip_demoed_instances,
                    )
                )
                self.verbalization.steps.append(
                    Set(
                        fields={
                            "recipe_metadata/num_demos": self.num_demos,
                            "recipe_metadata/demos_pool_size": self.demos_pool_size,
                        }
                    )
                )

            elif isinstance(self.num_demos, list):
                self.verbalization.steps.append(
                    RandomSizeSample(
                        from_field=self.demos_pool_field_name,
                        to_field=self.demos_field,
                        sampler=self.sampler,
                        sample_sizes=self.num_demos,
                        skip_demoed_instances=self.skip_demoed_instances,
                    )
                )
                self.verbalization.steps.append(
                    GetLength(
                        field=constants.demos_field,
                        to_field="recipe_metadata/num_demos",
                    )
                )
                self.verbalization.steps.append(
                    Set(
                        fields={"recipe_metadata/demos_pool_size": self.demos_pool_size}
                    )
                )

            else:
                raise ValueError("num_demos must be int or List[int]")

            if isinstance(self.template, list):
                self.verbalization.steps.append(
                    ApplyRandomTemplate(
                        templates=self.template, demos_field=self.demos_field
                    )
                )
            else:
                self.verbalization.steps.append(
                    ApplySingleTemplate(
                        template=self.template, demos_field=self.demos_field
                    )
                )

        else:
            self.verbalization.steps.append(
                Set(
                    fields={
                        "recipe_metadata/num_demos": 0,
                        "recipe_metadata/demos_pool_size": 0,
                    }
                )
            )
            if isinstance(self.template, list):
                self.verbalization.steps.append(
                    ApplyRandomTemplate(templates=self.template)
                )
            else:
                self.verbalization.steps.append(
                    ApplySingleTemplate(template=self.template)
                )

        self.verbalization.steps.append(self.system_prompt)
        self.verbalization.steps.append(self.format)

        if self.postprocessors is not None:
            self.finalize.steps.append(
                Set(fields={"postprocessors": self.postprocessors})
            )

        if self.metrics is not None:
            self.finalize.steps.append(Set(fields={"metrics": self.metrics}))

        self.finalize.steps.append(FinalizeDataset(group_by=self.group_by))

    @property
    def has_card_templates(self):
        return (
            self.card is not None
            and self.card.templates is not None
            and len(self.card.templates) > 0
        )

    @property
    def has_no_templates(self):
        return self.template_card_index is None and self.template is None

    def prepare(self):
        assert (
            self.template_card_index is None or self.template is None
        ), f"Specify either template ({self.template}) or template_card_index ({self.template_card_index}) but not both"

        if self.has_no_templates:
            if self.has_card_templates:
                if isinstance(self.card.templates, list):
                    self.template_card_index = 0
                else:
                    self.template_card_index = next(iter(self.card.templates.keys()))
                logger.warning(
                    "Template was not specified in recipe, using the first template from the card by default."
                )
            else:
                self.template = self.card.task.default_template

        if self.template is None and self.template_card_index is not None:
            try:
                self.template = self.card.templates[self.template_card_index]
            except Exception as e:
                if isinstance(self.card.templates, dict):
                    options = list(self.card.templates.keys())
                else:
                    options = list(range(0, len(self.card.templates)))
                raise ValueError(
                    f"card_template_index '{self.template_card_index}' is not defined in card. Possible card_template_index options: {options}"
                ) from e

        if self.template is None:
            raise ValueError(
                "No template was specified in the the 'template' or 'template_card_index' recipe arguments, and no default templates are defined the card or task"
            )

        if self.use_demos:
            assert (
                self.demos_pool is not None
                and isoftype(self.demos_pool, List[Dict[str, Any]])
            ) != (
                self.demos_taken_from is not None
                and self.demos_pool_size is not None
                and self.demos_removed_from_data is not None
            ), (
                "The demos_pool must be specified by exactly one of two ways: explicitly, as a list of instances coming through parameter "
                + "'demos_pool', or via parameters 'demos_taken_from', 'demos_pool_size', and 'demos_removed_from_data', "
                + "that together direct its production."
            )

        # now set self.demos_pool_size for the checks done by verify
        if self.demos_pool:
            self.demos_pool_size = len(self.demos_pool)
        if self.demos_pool_size is not None and self.demos_pool_size == -1:
            self.demos_pool_size = sys.maxsize

        if isinstance(self.template, TemplatesList):
            self.template = self.template.items

        self.reset_pipeline()


@deprecation(version="2.0.0", alternative=DatasetRecipe)
class BaseRecipe(DatasetRecipe):
    pass


@deprecation(version="2.0.0", alternative=DatasetRecipe)
class StandardRecipeWithIndexes(DatasetRecipe):
    pass


@deprecation(version="2.0.0", alternative=DatasetRecipe)
class StandardRecipe(DatasetRecipe):
    pass
