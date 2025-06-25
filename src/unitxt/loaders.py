"""This section describes unitxt loaders.

Loaders: Generators of Unitxt Multistreams from existing date sources
=====================================================================

Unitxt is all about readily preparing of any given data source for feeding into any given language model, and then,
post-processing the model's output, preparing it for any given evaluator.

Through that journey, the data advances in the form of Unitxt Multistream, undergoing a sequential application
of various off-the-shelf operators (i.e., picked from Unitxt catalog), or operators easily implemented by inheriting.
The journey starts by a Unitxt Loader bearing a Multistream from the given datasource.
A loader, therefore, is the first item on any Unitxt Recipe.

Unitxt catalog contains several loaders for the most popular datasource formats.
All these loaders inherit from Loader, and hence, implementing a loader to expand over a new type of datasource is
straightforward.

Available Loaders Overview:
    - :class:`LoadHF <unitxt.loaders.LoadHF>` - Loads data from HuggingFace Datasets.
    - :class:`LoadCSV <unitxt.loaders.LoadCSV>` - Imports data from CSV (Comma-Separated Values) files.
    - :class:`LoadFromKaggle <unitxt.loaders.LoadFromKaggle>` - Retrieves datasets from the Kaggle community site.
    - :class:`LoadFromIBMCloud <unitxt.loaders.LoadFromIBMCloud>` - Fetches datasets hosted on IBM Cloud.
    - :class:`LoadFromSklearn <unitxt.loaders.LoadFromSklearn>` - Loads datasets available through the sklearn library.
    - :class:`MultipleSourceLoader <unitxt.loaders.MultipleSourceLoader>` - Combines data from multiple different sources.
    - :class:`LoadFromDictionary <unitxt.loaders.LoadFromDictionary>` - Loads data from a user-defined Python dictionary.
    - :class:`LoadFromHFSpace <unitxt.loaders.LoadFromHFSpace>` - Downloads and loads data from HuggingFace Spaces.




------------------------
"""

import fnmatch
import itertools
import json
import os
import tempfile
import time
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import pandas as pd
import requests
from datasets import (
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    get_dataset_split_names,
)
from datasets import load_dataset as _hf_load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from .dataclass import NonPositionalField
from .dict_utils import dict_get
from .error_utils import Documentation, UnitxtError, UnitxtWarning, error_context
from .fusion import FixedFusion
from .logging_utils import get_logger
from .operator import SourceOperator
from .operators import Set
from .settings_utils import get_settings
from .stream import DynamicStream, MultiStream
from .type_utils import isoftype
from .utils import LRUCache, recursive_copy, retry_connection_with_exponential_backoff

logger = get_logger()
settings = get_settings()


class UnitxtUnverifiedCodeError(UnitxtError):
    def __init__(self, path):
        super().__init__(
            f"Loader cannot load and run remote code from {path} in huggingface without setting unitxt.settings.allow_unverified_code=True or by setting environment variable: UNITXT_ALLOW_UNVERIFIED_CODE.",
            Documentation.SETTINGS,
        )


@retry_connection_with_exponential_backoff(backoff_factor=2)
def hf_load_dataset(path: str, *args, **kwargs):
    with error_context(
        stage="Raw Dataset Download",
        help="https://www.unitxt.ai/en/latest/unitxt.loaders.html#module-unitxt.loaders",
    ):
        if settings.hf_offline_datasets_path is not None:
            path = os.path.join(settings.hf_offline_datasets_path, path)
        try:
            return _hf_load_dataset(
                path,
                *args,
                **kwargs,
                verification_mode="no_checks",
                trust_remote_code=settings.allow_unverified_code,
                download_mode="force_redownload"
                if settings.disable_hf_datasets_cache
                else "reuse_dataset_if_exists",
            )
        except ValueError as e:
            if "trust_remote_code" in str(e):
                raise UnitxtUnverifiedCodeError(path) from e
            raise e  # Re raise


@retry_connection_with_exponential_backoff(backoff_factor=2)
def hf_get_dataset_splits(path: str, name: str, revision=None):
    try:
        return get_dataset_split_names(
            path=path,
            config_name=name,
            trust_remote_code=settings.allow_unverified_code,
            revision=revision,
        )
    except Exception as e:
        if "trust_remote_code" in str(e):
            raise UnitxtUnverifiedCodeError(path) from e

        if "Couldn't find cache" in str(e):
            raise FileNotFoundError(
                f"Dataset cache path={path}, name={name} was not found."
            ) from e
        raise e  # Re raise


class Loader(SourceOperator):
    """A base class for all loaders.

    A loader is the first component in the Unitxt Recipe,
    responsible for loading data from various sources and preparing it as a MultiStream for processing.
    The loader_limit is an optional parameter used to control the maximum number of instances to load from the data source.  It is applied for each split separately.
    It is usually provided to the loader via the recipe (see standard.py)
    The loader can use this value to limit the amount of data downloaded from the source
    to reduce loading time.  However, this may not always be possible, so the
    loader may ignore this.  In any case, the recipe, will limit the number of instances in the returned
    stream, after load is complete.

    Args:
        loader_limit: Optional integer to specify a limit on the number of records to load.
        streaming: Bool indicating if streaming should be used.
        num_proc: Optional integer to specify the number of processes to use for parallel dataset loading. Adjust the value according to the number of CPU cores available and the specific needs of your processing task.
    """

    loader_limit: int = None
    streaming: bool = False
    num_proc: int = None

    # class level shared cache:
    _loader_cache = LRUCache(max_size=settings.loader_cache_size)

    def get_limit(self) -> int:
        if settings.global_loader_limit is not None and self.loader_limit is not None:
            return min(int(settings.global_loader_limit), self.loader_limit)
        if settings.global_loader_limit is not None:
            return int(settings.global_loader_limit)
        return self.loader_limit

    def get_limiter(self):
        if settings.global_loader_limit is not None and self.loader_limit is not None:
            if int(settings.global_loader_limit) > self.loader_limit:
                return f"{self.__class__.__name__}.loader_limit"
            return "unitxt.settings.global_loader_limit"
        if settings.global_loader_limit is not None:
            return "unitxt.settings.global_loader_limit"
        return f"{self.__class__.__name__}.loader_limit"

    def log_limited_loading(self):
        if (
            not hasattr(self, "_already_logged_limited_loading")
            or not self._already_logged_limited_loading
        ):
            self._already_logged_limited_loading = True
            logger.info(
                f"\nLoading limited to {self.get_limit()} instances by setting {self.get_limiter()};"
            )

    def add_data_classification(self, multi_stream: MultiStream) -> MultiStream:
        if self.data_classification_policy is None:
            get_logger().warning(
                f"The {self.get_pretty_print_name()} loader does not set the `data_classification_policy`. "
                f"This may lead to sending of undesired data to external services.\n"
                f"Set it to a list of classification identifiers. \n"
                f"For example:\n"
                f"data_classification_policy = ['public']\n"
                f" or \n"
                f"data_classification_policy =['confidential','pii'])\n"
            )

        operator = Set(
            fields={"data_classification_policy": self.data_classification_policy}
        )
        return operator(multi_stream)

    def set_default_data_classification(
        self, default_data_classification_policy, additional_info
    ):
        if self.data_classification_policy is None:
            if additional_info is not None:
                logger.info(
                    f"{self.get_pretty_print_name()} sets 'data_classification_policy' to "
                    f"{default_data_classification_policy} by default {additional_info}.\n"
                    "To use a different value or remove this message, explicitly set the "
                    "`data_classification_policy` attribute of the loader.\n"
                )
            self.data_classification_policy = default_data_classification_policy

    @abstractmethod
    def load_iterables(self) -> Dict[str, Iterable]:
        pass

    def _maybe_set_classification_policy(self):
        pass

    def load_data(self) -> MultiStream:
        with error_context(
            self,
            stage="Data Loading",
            help="https://www.unitxt.ai/en/latest/unitxt.loaders.html#module-unitxt.loaders",
        ):
            iterables = self.load_iterables()
            if isoftype(iterables, MultiStream):
                return iterables
            return MultiStream.from_iterables(iterables, copying=True)

    def process(self) -> MultiStream:
        self._maybe_set_classification_policy()
        return self.add_data_classification(self.load_data())

    def get_splits(self):
        return list(self().keys())


class LazyLoader(Loader):
    split: Optional[str] = NonPositionalField(default=None)

    @abstractmethod
    def get_splits(self) -> List[str]:
        pass

    @abstractmethod
    def split_generator(self, split: str) -> Generator:
        pass

    def load_iterables(self) -> Union[Dict[str, DynamicStream], IterableDatasetDict]:
        if self.split is not None:
            splits = [self.split]
        else:
            splits = self.get_splits()

        return MultiStream(
            {
                split: DynamicStream(self.split_generator, gen_kwargs={"split": split})
                for split in splits
            }
        )


class LoadHF(LazyLoader):
    """Loads datasets from the HuggingFace Hub.

    It supports loading with or without streaming,
    and it can filter datasets upon loading.

    Args:
        path:
            The path or identifier of the dataset on the HuggingFace Hub.
        name:
            An optional dataset name.
        data_dir:
            Optional directory to store downloaded data.
        split:
            Optional specification of which split to load.
        data_files:
            Optional specification of particular data files to load. When you provide a list of data_files to Hugging Face's load_dataset function without explicitly specifying the split argument, these files are automatically placed into the train split.
        revision:
            Optional. The revision of the dataset. Often the commit id. Use in case you want to set the dataset version.
        streaming (bool):
            indicating if streaming should be used.
        filtering_lambda (str, optional):
            A lambda function for filtering the data after loading.
        num_proc (int, optional):
            Specifies the number of processes to use for parallel dataset loading.

    Example:
        Loading glue's mrpc dataset

        .. code-block:: python

            load_hf = LoadHF(path='glue', name='mrpc')
    """

    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    split: Optional[str] = None
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None
    revision: Optional[str] = None
    streaming: bool = None
    filtering_lambda: Optional[str] = None
    num_proc: Optional[int] = None
    splits: Optional[List[str]] = None

    def filter_load(self, dataset: DatasetDict):
        if not settings.allow_unverified_code:
            raise ValueError(
                f"{self.__class__.__name__} cannot run use filtering_lambda expression without setting unitxt.settings.allow_unverified_code=True or by setting environment variable: UNITXT_ALLOW_UNVERIFIED_CODE=True."
            )
        logger.info(f"\nLoading filtered by: {self.filtering_lambda};")
        return dataset.filter(eval(self.filtering_lambda))

    def is_streaming(self) -> bool:
        if self.streaming is None:
            return settings.stream_hf_datasets_by_default
        return self.streaming

    def is_in_cache(self, split):
        dataset_id = str(self) + "_" + str(split)
        return dataset_id in self.__class__._loader_cache

    # returns Dict when split names are not known in advance, and just the the single split dataset - if known
    def load_dataset(
        self, split: str, streaming=None, disable_memory_caching=False
    ) -> Union[IterableDatasetDict, IterableDataset]:
        dataset_id = str(self) + "_" + str(split)
        dataset = self.__class__._loader_cache.get(dataset_id, None)
        if dataset is None:
            if streaming is None:
                streaming = self.is_streaming()

            dataset = hf_load_dataset(
                self.path,
                name=self.name,
                data_dir=self.data_dir,
                data_files=self.data_files,
                revision=self.revision,
                streaming=streaming,
                split=split,
                num_proc=self.num_proc,
            )

            if dataset is None:
                raise NotImplementedError() from None

            if not disable_memory_caching:
                self.__class__._loader_cache.max_size = settings.loader_cache_size
                self.__class__._loader_cache[dataset_id] = dataset
        self._already_logged_limited_loading = True

        return dataset

    def _maybe_set_classification_policy(self):
        if os.path.exists(self.path):
            self.set_default_data_classification(
                ["proprietary"], "when loading from local files"
            )
        else:
            self.set_default_data_classification(
                ["public"],
                None,  # No warning when loading from public hub
            )

    @retry_connection_with_exponential_backoff(max_retries=3, backoff_factor=2)
    def get_splits(self):
        if self.splits is not None:
            return self.splits
        if self.data_files is not None:
            if isinstance(self.data_files, dict):
                return list(self.data_files.keys())
            return ["train"]
        try:
            return hf_get_dataset_splits(
                path=self.path,
                name=self.name,
                revision=self.revision,
            )
        except Exception:
            UnitxtWarning(
                f'LoadHF(path="{self.path}", name="{self.name}") could not retrieve split names without loading the dataset. Consider defining "splits" in the LoadHF definition to improve loading time.'
            )
            try:
                dataset = self.load_dataset(
                    split=None, disable_memory_caching=True, streaming=True
                )
            except NotImplementedError:  # streaming is not supported for zipped files so we load without streaming
                dataset = self.load_dataset(split=None, streaming=False)

            if dataset is None:
                raise FileNotFoundError(
                    f"Dataset path={self.path}, name={self.name} was not found."
                ) from None

            return list(dataset.keys())

    def split_generator(self, split: str) -> Generator:
        if self.get_limit() is not None:
            if not self.is_in_cache(split):
                self.log_limited_loading()
        try:
            dataset = self.load_dataset(split=split)
        except (
            NotImplementedError
        ):  # streaming is not supported for zipped files so we load without streaming
            dataset = self.load_dataset(split=split, streaming=False)

        if self.filtering_lambda is not None:
            dataset = self.filter_load(dataset)

        limit = self.get_limit()
        if limit is None:
            yield from dataset
        else:
            for i, instance in enumerate(dataset):
                yield instance
                if i + 1 >= limit:
                    break


class LoadWithPandas(LazyLoader):
    """Utility base class for classes loading with pandas."""

    files: Dict[str, str]
    chunksize: int = 1000
    loader_limit: Optional[int] = None
    streaming: bool = True
    compression: Optional[str] = None

    def _maybe_set_classification_policy(self):
        self.set_default_data_classification(
            ["proprietary"], "when loading from local files"
        )

    def split_generator(self, split: str) -> Generator:
        dataset_id = str(self) + "_" + split
        dataset = self.__class__._loader_cache.get(dataset_id, None)
        if dataset is None:
            if self.get_limit() is not None:
                self.log_limited_loading()
            for attempt in range(settings.loaders_max_retries):
                try:
                    file = self.files[split]
                    if self.get_limit() is not None:
                        self.log_limited_loading()

                    try:
                        dataframe = self.read_dataframe(file)
                        break
                    except ValueError:
                        import fsspec

                        with fsspec.open(file, mode="rt") as file:
                            dataframe = self.read_dataframe(file)
                        break
                except Exception as e:
                    logger.warning(f"Attempt  load {attempt + 1} failed: {e}")
                    if attempt < settings.loaders_max_retries - 1:
                        time.sleep(2)
                    else:
                        raise e

            limit = self.get_limit()
            if limit is not None and len(dataframe) > limit:
                dataframe = dataframe.head(limit)

            dataset = dataframe.to_dict("records")

            self.__class__._loader_cache.max_size = settings.loader_cache_size
            self.__class__._loader_cache[dataset_id] = dataset

        for instance in self.__class__._loader_cache[dataset_id]:
            yield recursive_copy(instance)

    def get_splits(self) -> List[str]:
        return list(self.files.keys())

    def get_args(self) -> Dict[str, Any]:
        args = {}
        if self.compression is not None:
            args["compression"] = self.compression
        if self.get_limit() is not None:
            args["nrows"] = self.get_limit()
        return args

    @abstractmethod
    def read_dataframe(self, file) -> pd.DataFrame:
        ...


class LoadCSV(LoadWithPandas):
    """Loads data from CSV files.

    Supports streaming and can handle large files by loading them in chunks.

    Args:
        files (Dict[str, str]): A dictionary mapping names to file paths.
        chunksize : Size of the chunks to load at a time.
        loader_limit: Optional integer to specify a limit on the number of records to load.
        streaming: Bool indicating if streaming should be used.
        sep: String specifying the separator used in the CSV files.

    Example:
        Loading csv

        .. code-block:: python

            load_csv = LoadCSV(files={'train': 'path/to/train.csv'}, chunksize=100)
    """

    sep: str = ","

    def read_dataframe(self, file) -> pd.DataFrame:
        with error_context(
            stage="Raw Dataset Loading",
            help="https://www.unitxt.ai/en/latest/unitxt.loaders.html#module-unitxt.loaders",
        ):
            return pd.read_csv(
                file, sep=self.sep, low_memory=self.streaming, **self.get_args()
            )


def read_file(source) -> bytes:
    if hasattr(source, "read"):
        return source.read()

    if isinstance(source, str) and (
        source.startswith("http://") or source.startswith("https://")
    ):
        from urllib import request

        with request.urlopen(source) as response:
            return response.read()

    with open(source, "rb") as f:
        return f.read()


class LoadJsonFile(LoadWithPandas):
    """Loads data from JSON files.

    Supports streaming and can handle large files by loading them in chunks.

    Args:
        files (Dict[str, str]): A dictionary mapping names to file paths.
        chunksize : Size of the chunks to load at a time.
        loader_limit: Optional integer to specify a limit on the number of records to load.
        streaming: Bool indicating if streaming should be used.
        lines: Bool indicate if it is json lines file structure. Otherwise, assumes a single json object in the file.
        data_field: optional field within the json object, that contains the list of instances.

    Example:
        Loading json lines

        .. code-block:: python

            load_csv = LoadJsonFile(files={'train': 'path/to/train.jsonl'}, line=True, chunksize=100)
    """

    lines: bool = False
    data_field: Optional[str] = None

    def read_dataframe(self, file) -> pd.DataFrame:
        with error_context(
            stage="Raw Dataset Loading",
            help="https://www.unitxt.ai/en/latest/unitxt.loaders.html#module-unitxt.loaders",
        ):
            args = self.get_args()
            if not self.lines:
                data = json.loads(read_file(file))
                if self.data_field:
                    instances = dict_get(data, self.data_field)
                    if not isoftype(instances, List[Dict[str, Any]]):
                        raise UnitxtError(
                            f"{self.data_field} of file {file} is not a list of dictionariess in LoadJsonFile loader"
                        )
                else:
                    if isoftype(data, Dict[str, Any]):
                        instances = [data]
                    elif isoftype(data, List[Dict[str, Any]]):
                        instances = data
                    else:
                        raise UnitxtError(
                            f"data of file {file} is not dictionary or a list of dictionaries in LoadJsonFile loader"
                        )
                dataframe = pd.DataFrame(instances)
            else:
                if self.data_field is not None:
                    raise UnitxtError(
                        "Can not load from a specific 'data_field' when loading multiple lines (lines=True)"
                    )
                dataframe = pd.read_json(file, lines=self.lines, **args)
            return dataframe


class LoadFromSklearn(LazyLoader):
    """Loads datasets from the sklearn library.

    This loader does not support streaming and is intended for use with sklearn's dataset fetch functions.

    Args:
        dataset_name: The name of the sklearn dataset to fetch.
        splits: A list of data splits to load, e.g., ['train', 'test'].

    Example:
        Loading form sklearn

        .. code-block:: python

            load_sklearn = LoadFromSklearn(dataset_name='iris', splits=['train', 'test'])
    """

    dataset_name: str
    splits: List[str] = ["train", "test"]

    _requirements_list: List[str] = ["scikit-learn", "pandas"]

    data_classification_policy = ["public"]

    def verify(self):
        super().verify()

        if self.streaming:
            raise NotImplementedError("LoadFromSklearn cannot load with streaming.")

    def prepare(self):
        super().prepare()
        from sklearn import datasets as sklearn_datatasets

        self.downloader = getattr(sklearn_datatasets, f"fetch_{self.dataset_name}")

    def get_splits(self):
        return self.splits

    def split_generator(self, split: str) -> Generator:
        dataset_id = str(self) + "_" + split
        dataset = self.__class__._loader_cache.get(dataset_id, None)
        if dataset is None:
            with error_context(
                stage="Raw Dataset Loading",
                help="https://www.unitxt.ai/en/latest/unitxt.loaders.html#module-unitxt.loaders",
            ):
                split_data = self.downloader(subset=split)
                targets = [split_data["target_names"][t] for t in split_data["target"]]
            df = pd.DataFrame([split_data["data"], targets]).T
            df.columns = ["data", "target"]
            dataset = df.to_dict("records")
            self.__class__._loader_cache.max_size = settings.loader_cache_size
            self.__class__._loader_cache[dataset_id] = dataset
        for instance in self.__class__._loader_cache[dataset_id]:
            yield recursive_copy(instance)


class MissingKaggleCredentialsError(ValueError):
    pass


class LoadFromKaggle(Loader):
    """Loads datasets from Kaggle.

    Requires Kaggle API credentials and does not support streaming.

    Args:
        url: URL to the Kaggle dataset.

    Example:
        Loading from kaggle

        .. code-block:: python

            load_kaggle = LoadFromKaggle(url='kaggle.com/dataset/example')
    """

    url: str

    _requirements_list: List[str] = ["opendatasets"]
    data_classification_policy = ["public"]

    def verify(self):
        super().verify()
        if not os.path.isfile("kaggle.json"):
            raise MissingKaggleCredentialsError(
                "Please obtain kaggle credentials https://christianjmills.com/posts/kaggle-obtain-api-key-tutorial/ and save them to local ./kaggle.json file"
            )

        if self.streaming:
            raise NotImplementedError("LoadFromKaggle cannot load with streaming.")

    def prepare(self):
        super().prepare()
        from opendatasets import download

        self.downloader = download

    def load_iterables(self):
        with TemporaryDirectory() as temp_directory:
            self.downloader(self.url, temp_directory)
            return hf_load_dataset(temp_directory, streaming=False)


class LoadFromIBMCloud(Loader):
    """Loads data from IBM Cloud Object Storage.

    Does not support streaming and requires AWS-style access keys.
    data_dir Can be either:
    1. a list of file names, the split of each file is determined by the file name pattern
    2. Mapping: split -> file_name, e.g. {"test" : "test.json", "train": "train.json"}
    3. Mapping: split -> file_names, e.g. {"test" : ["test1.json", "test2.json"], "train": ["train.json"]}

    Args:
        endpoint_url_env:
            Environment variable name for the IBM Cloud endpoint URL.
        aws_access_key_id_env:
            Environment variable name for the AWS access key ID.
        aws_secret_access_key_env:
            Environment variable name for the AWS secret access key.
        bucket_name:
            Name of the S3 bucket from which to load data.
        data_dir:
            Optional directory path within the bucket.
        data_files:
            Union type allowing either a list of file names or a mapping of splits to file names.
        data_field:
            The dataset key for nested JSON file, i.e. when multiple datasets are nested in the same file
        caching (bool):
            indicating if caching is enabled to avoid re-downloading data.

    Example:
        Loading from IBM Cloud

        .. code-block:: python

            load_ibm_cloud = LoadFromIBMCloud(
                endpoint_url_env='IBM_CLOUD_ENDPOINT',
                aws_access_key_id_env='IBM_AWS_ACCESS_KEY_ID',
                aws_secret_access_key_env='IBM_AWS_SECRET_ACCESS_KEY', # pragma: allowlist secret
                bucket_name='my-bucket'
            )
            multi_stream = load_ibm_cloud.process()
    """

    endpoint_url_env: str
    aws_access_key_id_env: str
    aws_secret_access_key_env: str
    bucket_name: str
    data_dir: str = None

    data_files: Union[Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    data_field: str = None
    caching: bool = True
    data_classification_policy = ["proprietary"]

    _requirements_list: List[str] = ["ibm-cos-sdk"]

    def _download_from_cos(self, cos, bucket_name, item_name, local_file):
        logger.info(f"Downloading {item_name} from {bucket_name} COS")
        try:
            response = cos.Object(bucket_name, item_name).get()
            size = response["ContentLength"]
            body = response["Body"]
        except Exception as e:
            raise Exception(
                f"Unabled to access {item_name} in {bucket_name} in COS", e
            ) from e

        if self.get_limit() is not None:
            if item_name.endswith(".jsonl"):
                first_lines = list(
                    itertools.islice(body.iter_lines(), self.get_limit())
                )
                with open(local_file, "wb") as downloaded_file:
                    for line in first_lines:
                        downloaded_file.write(line)
                        downloaded_file.write(b"\n")
                logger.info(
                    f"\nDownload successful limited to {self.get_limit()} lines"
                )
                return

        progress_bar = tqdm(total=size, unit="iB", unit_scale=True)

        def upload_progress(chunk):
            progress_bar.update(chunk)

        try:
            cos.Bucket(bucket_name).download_file(
                item_name, local_file, Callback=upload_progress
            )
            logger.info("\nDownload Successful")
        except Exception as e:
            raise Exception(
                f"Unabled to download {item_name} in {bucket_name}", e
            ) from e

    def prepare(self):
        super().prepare()
        self.endpoint_url = os.getenv(self.endpoint_url_env)
        self.aws_access_key_id = os.getenv(self.aws_access_key_id_env)
        self.aws_secret_access_key = os.getenv(self.aws_secret_access_key_env)
        root_dir = os.getenv("UNITXT_IBM_COS_CACHE", None) or os.getcwd()
        self.cache_dir = os.path.join(root_dir, "ibmcos_datasets")

        if not os.path.exists(self.cache_dir):
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.verified = False

    def lazy_verify(self):
        super().verify()
        assert (
            self.endpoint_url is not None
        ), f"Please set the {self.endpoint_url_env} environmental variable"
        assert (
            self.aws_access_key_id is not None
        ), f"Please set {self.aws_access_key_id_env} environmental variable"
        assert (
            self.aws_secret_access_key is not None
        ), f"Please set {self.aws_secret_access_key_env} environmental variable"
        if self.streaming:
            raise NotImplementedError("LoadFromKaggle cannot load with streaming.")

    def _maybe_set_classification_policy(self):
        self.set_default_data_classification(
            ["proprietary"], "when loading from IBM COS"
        )

    def load_iterables(self):
        if not self.verified:
            self.lazy_verify()
            self.verified = True
        import ibm_boto3

        cos = ibm_boto3.resource(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
        )
        local_dir = os.path.join(
            self.cache_dir,
            self.bucket_name,
            self.data_dir or "",  # data_dir can be None
            f"loader_limit_{self.get_limit()}",
        )
        if not os.path.exists(local_dir):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
        if isinstance(self.data_files, Mapping):
            data_files_names = list(self.data_files.values())
            if not isinstance(data_files_names[0], str):
                data_files_names = list(itertools.chain(*data_files_names))
        else:
            data_files_names = self.data_files

        for data_file in data_files_names:
            local_file = os.path.join(local_dir, data_file)
            if not self.caching or not os.path.exists(local_file):
                # Build object key based on parameters. Slash character is not
                # allowed to be part of object key in IBM COS.
                object_key = (
                    self.data_dir + "/" + data_file
                    if self.data_dir is not None
                    else data_file
                )
                with error_context(
                    stage="Raw Dataset Download",
                    help="https://www.unitxt.ai/en/latest/unitxt.loaders.html#module-unitxt.loaders",
                ):
                    with tempfile.NamedTemporaryFile() as temp_file:
                        # Download to  a temporary file in same file partition, and then do an atomic move
                        self._download_from_cos(
                            cos,
                            self.bucket_name,
                            object_key,
                            local_dir + "/" + os.path.basename(temp_file.name),
                        )
                        os.renames(
                            local_dir + "/" + os.path.basename(temp_file.name),
                            local_dir + "/" + data_file,
                        )

        if isinstance(self.data_files, list):
            dataset = hf_load_dataset(local_dir, streaming=False, field=self.data_field)
        else:
            dataset = hf_load_dataset(
                local_dir,
                streaming=False,
                data_files=self.data_files,
                field=self.data_field,
            )

        return dataset


class MultipleSourceLoader(LazyLoader):
    """Allows loading data from multiple sources, potentially mixing different types of loaders.

    Args:
        sources: A list of loaders that will be combined to form a unified dataset.

    Examples:
        1) Loading the train split from a HuggingFace Hub and the test set from a local file:

        .. code-block:: python

            MultipleSourceLoader(sources = [ LoadHF(path="public/data",split="train"), LoadCSV({"test": "mytest.csv"}) ])



        2) Loading a test set combined from two files

        .. code-block:: python

            MultipleSourceLoader(sources = [ LoadCSV({"test": "mytest1.csv"}, LoadCSV({"test": "mytest2.csv"}) ])
    """

    sources: List[Loader]

    def add_data_classification(self, multi_stream: MultiStream) -> MultiStream:
        if self.data_classification_policy is None:
            return multi_stream
        return super().add_data_classification(multi_stream)

    def get_splits(self):
        splits = []
        for loader in self.sources:
            splits.extend(loader.get_splits())
        return list(set(splits))

    def split_generator(self, split: str) -> Generator[Any, None, None]:
        yield from FixedFusion(
            subsets=self.sources,
            max_instances_per_subset=self.get_limit(),
            include_splits=[split],
        )()[split]


class LoadFromDictionary(Loader):
    """Allows loading data from a dictionary of constants.

    The loader can be used, for example, when debugging or working with small datasets.

    Args:
        data (Dict[str, List[Dict[str, Any]]]): a dictionary of constants from which the data will be loaded

    Example:
        Loading dictionary

        .. code-block:: python

            data = {
                "train": [{"input": "SomeInput1", "output": "SomeResult1"},
                          {"input": "SomeInput2", "output": "SomeResult2"}],
                "test":  [{"input": "SomeInput3", "output": "SomeResult3"},
                          {"input": "SomeInput4", "output": "SomeResult4"}]
            }
            loader = LoadFromDictionary(data=data)
    """

    data: Dict[str, List[Dict[str, Any]]]

    def verify(self):
        super().verify()
        with error_context(
            stage="Dataset Loading",
            help="https://www.unitxt.ai/en/latest/unitxt.loaders.html#module-unitxt.loaders",
        ):
            if not isoftype(self.data, Dict[str, List[Dict[str, Any]]]):
                raise ValueError(
                    f"Passed data to LoadFromDictionary is not of type Dict[str, List[Dict[str, Any]]].\n"
                    f"Expected data should map between split name and list of instances.\n"
                    f"Received value: {self.data}\n"
                )
            for split in self.data.keys():
                if len(self.data[split]) == 0:
                    raise ValueError(f"Split {split} has no instances.")
                first_instance = self.data[split][0]
                for instance in self.data[split]:
                    if instance.keys() != first_instance.keys():
                        raise ValueError(
                            f"Not all instances in split '{split}' have the same fields.\n"
                            f"instance {instance} has different fields different from {first_instance}"
                        )

    def _maybe_set_classification_policy(self):
        self.set_default_data_classification(
            ["proprietary"], "when loading from python dictionary"
        )

    def load_iterables(self) -> MultiStream:
        return self.data


class LoadFromHFSpace(LazyLoader):
    """Used to load data from HuggingFace Spaces lazily.

    Args:
        space_name (str):
            Name of the HuggingFace Space to be accessed.
        data_files (str | Sequence[str] | Mapping[str, str | Sequence[str]]):
            Relative paths to files within a given repository. If given as a mapping,
            paths should be values, while keys should represent the type of respective files
            (training, testing etc.).
        path (str, optional):
            Absolute path to a directory where data should be downloaded.
        revision (str, optional):
            ID of a Git branch or commit to be used. By default, it is set to None,
            thus data is downloaded from the main branch of the accessed repository.
        use_token (bool, optional):
            Whether a token is used for authentication when accessing
            the HuggingFace Space. If necessary, the token is read from the HuggingFace
            config folder.
        token_env (str, optional):
            Key of an env variable which value will be used for
            authentication when accessing the HuggingFace Space - if necessary.
    """

    space_name: str
    data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    path: Optional[str] = None
    revision: Optional[str] = None
    use_token: Optional[bool] = None
    token_env: Optional[str] = None
    requirements_list: List[str] = ["huggingface_hub"]

    streaming: bool = True

    def _get_token(self) -> Optional[Union[bool, str]]:
        if self.token_env:
            token = os.getenv(self.token_env)
            if not token:
                get_logger().warning(
                    f"The 'token_env' parameter was specified as '{self.token_env}', "
                    f"however, no environment variable under such a name was found. "
                    f"Therefore, the loader will not use any tokens for authentication."
                )
            return token
        return self.use_token

    @staticmethod
    def _is_wildcard(path: str) -> bool:
        wildcard_characters = ["*", "?", "[", "]"]
        return any(char in path for char in wildcard_characters)

    def _get_repo_files(self):
        if not hasattr(self, "_repo_files") or self._repo_files is None:
            api = HfApi()
            self._repo_files = api.list_repo_files(
                self.space_name, repo_type="space", revision=self.revision
            )
        return self._repo_files

    def _get_sub_files(self, file: str) -> List[str]:
        if self._is_wildcard(file):
            return fnmatch.filter(self._get_repo_files(), file)
        return [file]

    def get_splits(self) -> List[str]:
        if isinstance(self.data_files, Mapping):
            return list(self.data_files.keys())
        return ["train"]  # Default to 'train' if not specified

    def split_generator(self, split: str) -> Generator:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        token = self._get_token()
        files = (
            self.data_files.get(split, self.data_files)
            if isinstance(self.data_files, Mapping)
            else self.data_files
        )

        if isinstance(files, str):
            files = [files]
        limit = self.get_limit()

        if limit is not None:
            total = 0
            self.log_limited_loading()

        for file in files:
            for sub_file in self._get_sub_files(file):
                try:
                    file_path = hf_hub_download(
                        repo_id=self.space_name,
                        filename=sub_file,
                        repo_type="space",
                        token=token,
                        revision=self.revision,
                        local_dir=self.path,
                    )
                except EntryNotFoundError as e:
                    raise ValueError(
                        f"The file '{file}' was not found in the space '{self.space_name}'. "
                        f"Please check if the filename is correct, or if it exists in that "
                        f"Huggingface space."
                    ) from e
                except RepositoryNotFoundError as e:
                    raise ValueError(
                        f"The Huggingface space '{self.space_name}' was not found. "
                        f"Please check if the name is correct and you have access to the space."
                    ) from e

                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        yield json.loads(line.strip())
                        if limit is not None:
                            total += 1
                            if total >= limit:
                                return


class LoadFromAPI(Loader):
    """Loads data from from API.

    This loader is designed to fetch data from an API endpoint,
    handling authentication through an API key. It supports
    customizable chunk sizes and limits for data retrieval.

    Args:
        urls (Dict[str, str]):
            A dictionary mapping split names to their respective API URLs.
        chunksize (int, optional):
            The size of data chunks to fetch in each request. Defaults to 100,000.
        loader_limit (int, optional):
            Limits the number of records to load. Applied per split. Defaults to None.
        streaming (bool, optional):
            Determines if data should be streamed. Defaults to False.
        api_key_env_var (str, optional):
            The name of the environment variable holding the API key.
            Defaults to "SQL_API_KEY".
        headers (Dict[str, Any], optional):
            Additional headers to include in API requests. Defaults to None.
        data_field (str, optional):
            The name of the field in the API response that contains the data.
            Defaults to "data".
        method (str, optional):
            The HTTP method to use for API requests. Defaults to "GET".
        verify_cert (bool):
            Apply verification of the SSL certificate
            Defaults as True
    """

    urls: Dict[str, str]
    chunksize: int = 100000
    loader_limit: Optional[int] = None
    streaming: bool = False
    api_key_env_var: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    data_field: str = "data"
    method: str = "GET"
    verify_cert: bool = True

    # class level shared cache:
    _loader_cache = LRUCache(max_size=settings.loader_cache_size)

    def _maybe_set_classification_policy(self):
        self.set_default_data_classification(["proprietary"], "when loading from API")

    def load_iterables(self) -> Dict[str, Iterable]:
        if self.api_key_env_var is not None:
            api_key = os.getenv(self.api_key_env_var, None)
            if not api_key:
                raise ValueError(
                    f"The environment variable '{self.api_key_env_var}' must be set to use the LoadFromAPI loader."
                )
        else:
            api_key = None

        base_headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        if api_key is not None:
            base_headers["Authorization"] = f"Bearer {api_key}"

        if self.headers:
            base_headers.update(self.headers)

        iterables = {}
        for split_name, url in self.urls.items():
            if self.get_limit() is not None:
                self.log_limited_loading()

            if self.method == "GET":
                response = requests.get(
                    url,
                    headers=base_headers,
                    verify=self.verify_cert,
                )
            elif self.method == "POST":
                response = requests.post(
                    url,
                    headers=base_headers,
                    verify=self.verify_cert,
                    json={},
                )
            else:
                raise ValueError(f"Method {self.method} not supported")

            response.raise_for_status()

            data = json.loads(response.text)

            if self.data_field:
                if self.data_field not in data:
                    raise ValueError(
                        f"Data field '{self.data_field}' not found in API response."
                    )
                data = data[self.data_field]

            if self.get_limit() is not None:
                data = data[: self.get_limit()]

            iterables[split_name] = data

        return iterables

    def process(self) -> MultiStream:
        self._maybe_set_classification_policy()
        iterables = self.__class__._loader_cache.get(str(self), None)
        if iterables is None:
            iterables = self.load_iterables()
            self.__class__._loader_cache.max_size = settings.loader_cache_size
            self.__class__._loader_cache[str(self)] = iterables
        return MultiStream.from_iterables(iterables, copying=True)
