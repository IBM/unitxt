import hashlib
import inspect
import json
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets.exceptions import DatasetGenerationError

from .artifact import fetch_artifact
from .benchmark import Benchmark
from .card import TaskCard
from .dataset_utils import get_dataset_artifact
from .error_utils import UnitxtError
from .inference import (
    InferenceEngine,
    LogProbInferenceEngine,
    OptionSelectingByLogProbsInferenceEngine,
)
from .loaders import LoadFromDictionary
from .logging_utils import get_logger
from .metric_utils import EvaluationResults, _compute, _inference_post_process
from .operator import SourceOperator
from .schema import loads_batch
from .settings_utils import get_constants, get_settings
from .standard import DatasetRecipe
from .task import Task

logger = get_logger()
constants = get_constants()
settings = get_settings()


def short_hex_hash(value, length=8):
    h = hashlib.sha256(value.encode()).hexdigest()  # Full 64-character hex
    return h[:length]


def _get_recipe_from_query(
    dataset_query: str, overwrite_kwargs: Optional[Dict[str, Any]] = None
) -> DatasetRecipe:
    try:
        dataset_stream, _ = fetch_artifact(
            dataset_query, overwrite_kwargs=overwrite_kwargs
        )
    except:
        dataset_stream = get_dataset_artifact(
            dataset_query, overwrite_kwargs=overwrite_kwargs
        )
    return dataset_stream


def _get_recipe_from_dict(dataset_params: Dict[str, Any]) -> DatasetRecipe:
    recipe_attributes = list(DatasetRecipe.__dict__["__fields__"].keys())
    for param in dataset_params.keys():
        assert param in recipe_attributes, (
            f"The parameter '{param}' is not an attribute of the 'DatasetRecipe' class. "
            f"Please check if the name is correct. The available attributes are: '{recipe_attributes}'."
        )
    return DatasetRecipe(**dataset_params)


def _verify_dataset_args(dataset_query: Optional[str] = None, dataset_args=None):
    if dataset_query and dataset_args:
        raise ValueError(
            "Cannot provide 'dataset_query' and key-worded arguments at the same time. "
            "If you want to load dataset from a card in local catalog, use query only. "
            "Otherwise, use key-worded arguments only to specify properties of dataset."
        )

    if dataset_query:
        if not isinstance(dataset_query, str):
            raise ValueError(
                f"If specified, 'dataset_query' must be a string, however, "
                f"'{dataset_query}' was provided instead, which is of type "
                f"'{type(dataset_query)}'."
            )

    if not dataset_query and not dataset_args:
        raise ValueError(
            "Either 'dataset_query' or key-worded arguments must be provided."
        )


def load_recipe(dataset_query: Optional[str] = None, **kwargs) -> DatasetRecipe:
    if isinstance(dataset_query, (DatasetRecipe, Benchmark)):
        return dataset_query

    if dataset_query:
        recipe = _get_recipe_from_query(dataset_query, kwargs)

    elif kwargs:
        recipe = _get_recipe_from_dict(kwargs)

    else:
        raise UnitxtError(
            "Specify either dataset recipe string artifact name or recipe args."
        )

    return recipe


def create_dataset(
    task: Union[str, Task],
    test_set: List[Dict[Any, Any]],
    train_set: Optional[List[Dict[Any, Any]]] = None,
    validation_set: Optional[List[Dict[Any, Any]]] = None,
    split: Optional[str] = None,
    data_classification_policy: Optional[List[str]] = None,
    **kwargs,
) -> Union[DatasetDict, IterableDatasetDict, Dataset, IterableDataset]:
    """Creates dataset from input data based on a specific task.

    Args:
        task:  The name of the task from the Unitxt Catalog (https://www.unitxt.ai/en/latest/catalog/catalog.tasks.__dir__.html)
        test_set : required list of instances
        train_set : optional train_set
        validation_set: optional validation set
        split: optional one split to choose
        data_classification_policy: data_classification_policy
        **kwargs: Arguments used to load dataset from provided datasets (see load_dataset())

    Returns:
        DatasetDict

    Example:
        template = Template(...)
        dataset = create_dataset(task="tasks.qa.open", template=template, format="formats.chatapi")
    """
    data = {"test": test_set}
    if train_set is not None:
        data["train"] = train_set
    if validation_set is not None:
        data["validation"] = validation_set
    task, _ = fetch_artifact(task)

    if "template" not in kwargs and task.default_template is None:
        raise Exception(
            f"No 'template' was passed to the create_dataset() and the given task ('{task.__id__}') has no 'default_template' field."
        )

    card = TaskCard(
        loader=LoadFromDictionary(
            data=data, data_classification_policy=data_classification_policy
        ),
        task=task,
    )
    return load_dataset(card=card, split=split, **kwargs)


def _source_to_dataset(
    source: SourceOperator,
    split=None,
    use_cache=False,
    streaming=False,
):
    from .dataset import Dataset as UnitxtDataset

    stream = source()

    try:
        ds_builder = UnitxtDataset(
            dataset_name="unitxt",
            config_name="recipe-" + short_hex_hash(repr(source)),
            version=constants.version,
        )
        if split is not None:
            stream = {split: stream[split]}
        ds_builder._generators = stream

        ds_builder.download_and_prepare(
            verification_mode="no_checks",
            download_mode=None if use_cache else "force_redownload",
        )

        if streaming:
            return ds_builder.as_streaming_dataset(split=split)

        return ds_builder.as_dataset(
            split=split, run_post_process=False, verification_mode="no_checks"
        )

    except DatasetGenerationError as e:
        raise e.__cause__


def load_dataset(
    dataset_query: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Union[DatasetDict, IterableDatasetDict, Dataset, IterableDataset]:
    """Loads dataset.

    If the 'dataset_query' argument is provided, then dataset is loaded from a card
    in local catalog based on parameters specified in the query.

    Alternatively, dataset is loaded from a provided card based on explicitly
    given parameters.

    If both are given, then the textual recipe is loaded with the key word args overriding the textual recipe args.

    Args:
        dataset_query (str, optional):
            A string query which specifies a dataset to load from
            local catalog or name of specific recipe or benchmark in the catalog. For
            example, ``"card=cards.wnli,template=templates.classification.multi_class.relation.default"``.
        streaming (bool, False):
            When True yields the data as a stream.
            This is useful when loading very large datasets.
            Loading datasets as streams avoid loading all the data to memory, but requires the dataset's loader to support streaming.
        split (str, optional):
            The split of the data to load
        use_cache (bool, optional):
            If set to True, the returned Huggingface dataset is cached on local disk such that if the same dataset is loaded again, it will be loaded from local disk, resulting in faster runs.
            If set to False (default), the returned dataset is not cached.
            Note that if caching is enabled and the dataset card definition is changed, the old version in the cache may be returned.
            Enable caching only if you are sure you are working with fixed Unitxt datasets and definitions (e.g. running using predefined datasets from the Unitxt catalog).
        **kwargs:
            Arguments used to load dataset from provided card, which is not present in local catalog.

    Returns:
        DatasetDict

    :Example:

        .. code-block:: python

            dataset = load_dataset(
                dataset_query="card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5"
            )  # card and template must be present in local catalog

            # or built programmatically
            card = TaskCard(...)
            template = Template(...)
            loader_limit = 10
            dataset = load_dataset(card=card, template=template, loader_limit=loader_limit)

    """
    recipe = load_recipe(dataset_query, **kwargs)

    dataset = _source_to_dataset(
        source=recipe, split=split, use_cache=use_cache, streaming=streaming
    )

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    all_kwargs = {key: values[key] for key in args if key != "kwargs"}
    all_kwargs.update(kwargs)
    metadata = fill_metadata(**all_kwargs)
    if isinstance(dataset, dict):
        for ds in dataset.values():
            ds.info.description = metadata.copy()
    else:
        dataset.info.description = metadata
    return dataset


def fill_metadata(**kwargs):
    metadata = kwargs.copy()
    metadata["unitxt_version"] = get_constants().version
    metadata["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return metadata


def evaluate(
    predictions,
    dataset: Union[Dataset, IterableDataset] = None,
    data=None,
    calc_confidence_intervals: bool = True,
) -> EvaluationResults:
    if dataset is None and data is None:
        raise UnitxtError(message="Specify 'dataset' in evaluate")
    if data is not None:
        dataset = data  # for backward compatibility
    evaluation_result = _compute(
        predictions=predictions,
        references=dataset,
        calc_confidence_intervals=calc_confidence_intervals,
    )
    if hasattr(dataset, "info") and hasattr(dataset.info, "description"):
        evaluation_result.metadata["dataset"] = dataset.info.description
    if hasattr(predictions, "metadata"):
        evaluation_result.metadata["predictions"] = predictions.metadata
    evaluation_result.metadata["creation_time"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]
    return evaluation_result


def post_process(predictions, data) -> List[Dict[str, Any]]:
    return _inference_post_process(predictions=predictions, references=data)


@lru_cache
def _get_produce_with_cache(dataset_query: Optional[str] = None, **kwargs):
    return load_recipe(dataset_query, **kwargs).produce


def produce(
    instance_or_instances, dataset_query: Optional[str] = None, **kwargs
) -> Union[Dataset, Dict[str, Any]]:
    is_list = isinstance(instance_or_instances, list)
    if not is_list:
        instance_or_instances = [instance_or_instances]
    result = _get_produce_with_cache(dataset_query, **kwargs)(instance_or_instances)
    if not is_list:
        return result[0]
    return Dataset.from_list(result).with_transform(loads_batch)


def infer(
    instance_or_instances,
    engine: InferenceEngine,
    dataset_query: Optional[str] = None,
    return_data: bool = False,
    return_log_probs: bool = False,
    return_meta_data: bool = False,
    previous_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
):
    dataset = produce(instance_or_instances, dataset_query, **kwargs)
    if previous_messages is not None:

        def add_previous_messages(example, index):
            example["source"] = previous_messages[index] + example["source"]
            return example

        dataset = dataset.map(add_previous_messages, with_indices=True)
    engine, _ = fetch_artifact(engine)
    if return_log_probs:
        if not isinstance(engine, LogProbInferenceEngine):
            raise NotImplementedError(
                f"Error in infer: return_log_probs set to True but supplied engine "
                f"{engine.__class__.__name__} does not support logprobs."
            )
        infer_outputs = engine.infer_log_probs(dataset, return_meta_data)
        raw_predictions = (
            [output.prediction for output in infer_outputs]
            if return_meta_data
            else infer_outputs
        )
        raw_predictions = [
            json.dumps(raw_prediction) for raw_prediction in raw_predictions
        ]
    else:
        infer_outputs = engine.infer(dataset, return_meta_data)
        raw_predictions = (
            [output.prediction for output in infer_outputs]
            if return_meta_data
            else infer_outputs
        )
    predictions = post_process(raw_predictions, dataset)
    if return_data:
        if return_meta_data:
            infer_output_list = [
                infer_output.__dict__ for infer_output in infer_outputs
            ]
            for infer_output in infer_output_list:
                del infer_output["prediction"]
            dataset = dataset.add_column("infer_meta_data", infer_output_list)
        dataset = dataset.add_column("prediction", predictions)
        return dataset.add_column("raw_prediction", raw_predictions)
    return predictions


def select(
    instance_or_instances,
    engine: OptionSelectingByLogProbsInferenceEngine,
    dataset_query: Optional[str] = None,
    return_data: bool = False,
    previous_messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
):
    dataset = produce(instance_or_instances, dataset_query, **kwargs)
    if previous_messages is not None:

        def add_previous_messages(example, index):
            example["source"] = previous_messages[index] + example["source"]
            return example

        dataset = dataset.map(add_previous_messages, with_indices=True)
    engine, _ = fetch_artifact(engine)
    predictions = engine.select(dataset)
    # predictions = post_process(raw_predictions, dataset)
    if return_data:
        return dataset.add_column("prediction", predictions)
    return predictions
