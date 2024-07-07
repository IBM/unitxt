"""This section describes unitxt loaders.

Loaders: Generators of Unitxt Multistreams from existing date sources
==============================================================

Unitxt is all about readily preparing of any given data source for feeding into any given language model, and then,
post-processing the model's output, preparing it for any given evaluator.

Through that journey, the data advances in the form of Unitxt Multistream, undergoing a sequential application
of various off the shelf operators (i.e, picked from Unitxt catalog), or operators easily implemented by inheriting.
The journey starts by a Unitxt Loeader bearing a Multistream from the given datasource.
A loader, therefore, is the first item on any Unitxt Recipe.

Unitxt catalog contains several loaders for the most popular datasource formats.
All these loaders inherit from Loader, and hence, implementing a loader to expand over a new type of datasource, is
straight forward.

Available Loaders Overview:
    - :ref:`LoadHF <unitxt.loaders.LoadHF>` - Loads data from Huggingface datasets.
    - :ref:`LoadCSV <unitxt.loaders.LoadCSV>` - Imports data from CSV (Comma-Separated Values) files.
    - :ref:`LoadFromKaggle <unitxt.loaders.LoadFromKaggle>` - Retrieves datasets from the Kaggle community site.
    - :ref:`LoadFromIBMCloud <unitxt.loaders.LoadFromIBMCloud>` - Fetches datasets hosted on IBM Cloud.
    - :ref:`LoadFromSklearn <unitxt.loaders.LoadFromSklearn>` - Loads datasets available through the sklearn library.
    - :ref:`MultipleSourceLoader <unitxt.loaders.MultipleSourceLoader>` - Combines data from multiple different sources.
    - :ref:`LoadFromDictionary <unitxt.loaders.LoadFromDictionary>` - Loads data from a user-defined Python dictionary.
    - :ref:`LoadFromHFSpace <unitxt.loaders.LoadFromHFSpace>` - Downloads and loads data from Huggingface Spaces.




------------------------
"""

import fnmatch
import itertools
import os
import tempfile
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
from datasets import load_dataset as hf_load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from .dataclass import InternalField, OptionalField
from .fusion import FixedFusion
from .logging_utils import get_logger
from .operator import SourceOperator
from .operators import Set
from .settings_utils import get_settings
from .stream import DynamicStream, MultiStream
from .type_utils import isoftype

logger = get_logger()
settings = get_settings()


class Loader(SourceOperator):
    """A base class for all loaders.

    A loader is the first component in the Unitxt Recipe,
    responsible for loading data from various sources and preparing it as a MultiStream for processing.
    The loader_limit an optional parameter used to control the maximum number of instances to load from the data source.  It is applied for each split separately.
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

    def get_limit(self):
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

    def sef_default_data_classification(
        self, default_data_classification_policy, additional_info
    ):
        if self.data_classification_policy is None:
            logger.info(
                f"{self.get_pretty_print_name()} sets 'data_classification_policy' to "
                f"{default_data_classification_policy} by default {additional_info}.\n"
                "To use a different value or remove this message, explicitly set the "
                "`data_classification_policy` attribute of the loader.\n"
            )
            self.data_classification_policy = default_data_classification_policy

    @abstractmethod
    def load_data(self):
        pass

    def process(self) -> MultiStream:
        return self.add_data_classification(self.load_data())


class LoadHF(Loader):
    """Loads datasets from the Huggingface Hub.

    It supports loading with or without streaming,
    and can filter datasets upon loading.

    Args:
        path: The path or identifier of the dataset on the Huggingface Hub.
        name: An optional dataset name.
        data_dir: Optional directory to store downloaded data.
        split: Optional specification of which split to load.
        data_files: Optional specification of particular data files to load.
        streaming: Bool indicating if streaming should be used.
        filtering_lambda: A lambda function for filtering the data after loading.
        num_proc: Optional integer to specify the number of processes to use for parallel dataset loading.

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
    streaming: bool = True
    filtering_lambda: Optional[str] = None
    num_proc: Optional[int] = None
    _cache: dict = InternalField(default=None)
    requirements_list: List[str] = OptionalField(default_factory=list)

    def verify(self):
        for requirement in self.requirements_list:
            if requirement not in self._requirements_list:
                self._requirements_list.append(requirement)
        super().verify()

    def filter_load(self, dataset):
        if not settings.allow_unverified_code:
            raise ValueError(
                f"{self.__class__.__name__} cannot run use filtering_lambda expression without setting unitxt.settings.allow_unverified_code=True or by setting environment variable: UNITXT_ALLOW_UNVERIFIED_CODE."
            )
        logger.info(f"\nLoading filtered by: {self.filtering_lambda};")
        return dataset.filter(eval(self.filtering_lambda))

    def stream_dataset(self):
        if self._cache is None:
            with tempfile.TemporaryDirectory() as dir_to_be_deleted:
                try:
                    dataset = hf_load_dataset(
                        self.path,
                        name=self.name,
                        data_dir=self.data_dir,
                        data_files=self.data_files,
                        streaming=self.streaming,
                        cache_dir=None if self.streaming else dir_to_be_deleted,
                        split=self.split,
                        trust_remote_code=settings.allow_unverified_code,
                        num_proc=self.num_proc,
                    )
                except ValueError as e:
                    if "trust_remote_code" in str(e):
                        raise ValueError(
                            f"{self.__class__.__name__} cannot run remote code from huggingface without setting unitxt.settings.allow_unverified_code=True or by setting environment variable: UNITXT_ALLOW_UNVERIFIED_CODE."
                        ) from e
                    raise e

            if self.split is not None:
                dataset = {self.split: dataset}

            self._cache = dataset

        else:
            dataset = self._cache

        if self.filtering_lambda is not None:
            dataset = self.filter_load(dataset)

        return dataset

    def load_dataset(self):
        if self._cache is None:
            with tempfile.TemporaryDirectory() as dir_to_be_deleted:
                try:
                    dataset = hf_load_dataset(
                        self.path,
                        name=self.name,
                        data_dir=self.data_dir,
                        data_files=self.data_files,
                        streaming=False,
                        keep_in_memory=True,
                        cache_dir=dir_to_be_deleted,
                        split=self.split,
                        trust_remote_code=settings.allow_unverified_code,
                        num_proc=self.num_proc,
                    )
                except ValueError as e:
                    if "trust_remote_code" in str(e):
                        raise ValueError(
                            f"{self.__class__.__name__} cannot run remote code from huggingface without setting unitxt.settings.allow_unverified_code=True or by setting environment variable: UNITXT_ALLOW_UNVERIFIED_CODE."
                        ) from e

            if self.split is None:
                for split in dataset.keys():
                    dataset[split] = dataset[split].to_iterable_dataset()
            else:
                dataset = {self.split: dataset}

            self._cache = dataset

        else:
            dataset = self._cache

        if self.filtering_lambda is not None:
            dataset = self.filter_load(dataset)

        return dataset

    def split_limited_load(self, dataset, split_name):
        yield from itertools.islice(dataset[split_name], self.get_limit())

    def limited_load(self, dataset):
        self.log_limited_loading()
        return MultiStream(
            {
                name: DynamicStream(
                    generator=self.split_limited_load,
                    gen_kwargs={"dataset": dataset, "split_name": name},
                )
                for name in self._cache.keys()
            }
        )

    def load_data(self):
        if os.path.exists(self.path):
            self.sef_default_data_classification(
                ["proprietary"], "when loading from local files"
            )
        else:
            self.sef_default_data_classification(
                ["public"], "when loading from Huggingface hub"
            )
        try:
            dataset = self.stream_dataset()
        except (
            NotImplementedError
        ):  # streaming is not supported for zipped files so we load without streaming
            dataset = self.load_dataset()

        if self.get_limit() is not None:
            return self.limited_load(dataset=dataset)

        return MultiStream.from_iterables(dataset)


class LoadCSV(Loader):
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

    files: Dict[str, str]
    chunksize: int = 1000
    _cache = InternalField(default_factory=dict)
    loader_limit: Optional[int] = None
    streaming: bool = True
    sep: str = ","

    def stream_csv(self, file):
        if self.get_limit() is not None:
            self.log_limited_loading()
            chunksize = min(self.get_limit(), self.chunksize)
        else:
            chunksize = self.chunksize

        row_count = 0
        for chunk in pd.read_csv(file, chunksize=chunksize, sep=self.sep):
            for _, row in chunk.iterrows():
                if self.get_limit() is not None and row_count >= self.get_limit():
                    return
                yield row.to_dict()
                row_count += 1

    def load_csv(self, file):
        if file not in self._cache:
            if self.get_limit() is not None:
                self.log_limited_loading()
                self._cache[file] = pd.read_csv(
                    file, nrows=self.get_limit(), sep=self.sep
                ).to_dict("records")
            else:
                self._cache[file] = pd.read_csv(file).to_dict("records")

        yield from self._cache[file]

    def load_data(self):
        self.sef_default_data_classification(
            ["proprietary"], "when loading from local files"
        )
        if self.streaming:
            return MultiStream(
                {
                    name: DynamicStream(
                        generator=self.stream_csv, gen_kwargs={"file": file}
                    )
                    for name, file in self.files.items()
                }
            )

        return MultiStream(
            {
                name: DynamicStream(generator=self.load_csv, gen_kwargs={"file": file})
                for name, file in self.files.items()
            }
        )


class LoadFromSklearn(Loader):
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

    _requirements_list: List[str] = ["sklearn", "pandas"]

    def verify(self):
        super().verify()

        if self.streaming:
            raise NotImplementedError("LoadFromSklearn cannot load with streaming.")

    def prepare(self):
        super().prepare()
        from sklearn import datasets as sklearn_datatasets

        self.downloader = getattr(sklearn_datatasets, f"fetch_{self.dataset_name}")

    def load_data(self):
        with TemporaryDirectory() as temp_directory:
            for split in self.splits:
                split_data = self.downloader(subset=split)
                targets = [split_data["target_names"][t] for t in split_data["target"]]
                df = pd.DataFrame([split_data["data"], targets]).T
                df.columns = ["data", "target"]
                df.to_csv(os.path.join(temp_directory, f"{split}.csv"), index=None)
            dataset = hf_load_dataset(temp_directory, streaming=False)

        return MultiStream.from_iterables(dataset)


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

    def load_data(self):
        with TemporaryDirectory() as temp_directory:
            self.downloader(self.url, temp_directory)
            dataset = hf_load_dataset(temp_directory, streaming=False)

        return MultiStream.from_iterables(dataset)


class LoadFromIBMCloud(Loader):
    """Loads data from IBM Cloud Object Storage.

    Does not support streaming and requires AWS-style access keys.
    data_dir Can be either:
    1. a list of file names, the split of each file is determined by the file name pattern
    2. Mapping: split -> file_name, e.g. {"test" : "test.json", "train": "train.json"}
    3. Mapping: split -> file_names, e.g. {"test" : ["test1.json", "test2.json"], "train": ["train.json"]}

    Args:
        endpoint_url_env: Environment variable name for the IBM Cloud endpoint URL.
        aws_access_key_id_env: Environment variable name for the AWS access key ID.
        aws_secret_access_key_env: Environment variable name for the AWS secret access key.
        bucket_name: Name of the S3 bucket from which to load data.
        data_dir: Optional directory path within the bucket.
        data_files: Union type allowing either a list of file names or a mapping of splits to file names.
        caching: Bool indicating if caching is enabled to avoid re-downloading data.

    Example:
        Loading from IBM Cloud

        .. code-block:: python

            load_ibm_cloud = LoadFromIBMCloud(
                endpoint_url_env='IBM_CLOUD_ENDPOINT',
                aws_access_key_id_env='IBM_AWS_ACCESS_KEY_ID',
                aws_secret_access_key_env='IBM_AWS_SECRET_ACCESS_KEY',
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
    caching: bool = True
    data_classification_policy = ["proprietary"]

    _requirements_list: List[str] = ["ibm_boto3"]

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

    def verify(self):
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

    def load_data(self):
        self.sef_default_data_classification(
            ["proprietary"], "when loading from IBM COS"
        )
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
            dataset = hf_load_dataset(local_dir, streaming=False)
        else:
            dataset = hf_load_dataset(
                local_dir, streaming=False, data_files=self.data_files
            )

        return MultiStream.from_iterables(dataset)


class MultipleSourceLoader(Loader):
    """Allows loading data from multiple sources, potentially mixing different types of loaders.

    Args:
        sources: A list of loaders that will be combined to form a unified dataset.

    Examples:
        1) Loading the train split from Huggingface hub and the test set from a local file:

        .. code-block:: python

            MultipleSourceLoader(loaders = [ LoadHF(path="public/data",split="train"), LoadCSV({"test": "mytest.csv"}) ])



        2) Loading a test set combined from two files

        .. code-block:: python

            MultipleSourceLoader(loaders = [ LoadCSV({"test": "mytest1.csv"}, LoadCSV({"test": "mytest2.csv"}) ])
    """

    sources: List[Loader]

    # MultipleSourceLoaders uses the the data classification from source loaders,
    # so only need to add it, if explicitly requested to override.
    def add_data_classification(self, multi_stream: MultiStream) -> MultiStream:
        if self.data_classification_policy is None:
            return multi_stream
        return super().add_data_classification(multi_stream)

    def load_data(self):
        return FixedFusion(
            origins=self.sources, max_instances_per_origin_split=self.get_limit()
        ).process()


class LoadFromDictionary(Loader):
    """Allows loading data from dictionary of constants.

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

    def load_data(self) -> MultiStream:
        self.sef_default_data_classification(
            ["proprietary"], "when loading from python dictionary"
        )
        return MultiStream.from_iterables(deepcopy(self.data))


class LoadFromHFSpace(LoadHF):
    """Used to load data from Huggingface spaces.

    Loaders firstly tries to download all files specified in the 'data_files' parameter
    from the given space and then reads them as a Huggingface dataset.

    Args:
        space_name (str): Name of the Huggingface space to be accessed to.
        data_files (str | Sequence[str] | Mapping[str, str | Sequence[str]]): Relative
            paths to files within a given repository. If given as a mapping, paths should
            be values, while keys should represent the type of respective files
            (training, testing etc.).
        path (str, optional): Absolute path to a directory where data should be downloaded to.
        revision (str, optional): ID of a Git branch or commit to be used. By default, it is
            set to None, thus data is downloaded from the main branch of the accessed
            repository.
        use_token (bool, optional): Whether token used for authentication when accessing
            the Huggingface space - if necessary - should be read from the Huggingface
            config folder.
        token_env (str, optional): Key of an env variable which value will be used for
            authentication when accessing the Huggingface space - if necessary.

    Example:
        Loading from Huggingface Space

        .. code-block:: python

            loader = LoadFromHFSpace(
                space_name="lmsys/mt-bench",
                data_files={
                    "train": [
                        "data/mt_bench/model_answer/gpt-3.5-turbo.jsonl",
                        "data/mt_bench/model_answer/gpt-4.jsonl",
                    ],
                    "test": "data/mt_bench/model_answer/tulu-30b.jsonl",
                },
            )
    """

    space_name: str
    data_files: Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    path: Optional[str] = None
    revision: Optional[str] = None
    use_token: Optional[bool] = None
    token_env: Optional[str] = None
    requirements_list: List[str] = ["huggingface_hub"]

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

    def _download_file_from_space(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

        token = self._get_token()

        try:
            file_path = hf_hub_download(
                repo_id=self.space_name,
                filename=filename,
                repo_type="space",
                token=token,
                revision=self.revision,
                local_dir=self.path,
            )
        except EntryNotFoundError as e:
            raise ValueError(
                f"The file '{filename}' was not found in the space '{self.space_name}'. "
                f"Please check if the filename is correct, or if it exists in that "
                f"Huggingface space."
            ) from e
        except RepositoryNotFoundError as e:
            raise ValueError(
                f"The Huggingface space '{self.space_name}' was not found. "
                f"Please check if the name is correct and you have access to the space."
            ) from e

        return file_path

    def _download_data(self) -> str:
        if isinstance(self.data_files, str):
            data_files = [self.data_files]
        elif isinstance(self.data_files, Mapping):
            data_files = list(self.data_files.values())
        else:
            data_files = self.data_files

        dir_paths_list = []
        for files in data_files:
            if isinstance(files, str):
                files = [files]

            paths = [self._download_file_from_space(file) for file in files]
            dir_paths = [
                path.replace(file_url, "") for path, file_url in zip(paths, files)
            ]
            dir_paths_list.extend(dir_paths)

        # All files - within the same space - are downloaded into the same base directory:
        assert len(set(dir_paths_list)) == 1

        return f"{dir_paths_list.pop()}"

    @staticmethod
    def _is_wildcard(path: str) -> bool:
        wildcard_characters = ["*", "?", "[", "]"]
        return any(char in path for char in wildcard_characters)

    def _get_file_list_from_wildcard_path(
        self, pattern: str, repo_files: List
    ) -> List[str]:
        if self._is_wildcard(pattern):
            return fnmatch.filter(repo_files, pattern)
        return [pattern]

    def _map_wildcard_path_to_full_paths(self):
        api = HfApi()
        repo_files = api.list_repo_files(self.space_name, repo_type="space")
        if isinstance(self.data_files, str):
            self.data_files = self._get_file_list_from_wildcard_path(
                self.data_files, repo_files
            )
        elif isinstance(self.data_files, Mapping):
            new_mapping = {}
            for k, v in self.data_files.items():
                if isinstance(v, list):
                    assert all(isinstance(s, str) for s in v)
                    new_mapping[k] = [
                        file
                        for p in v
                        for file in self._get_file_list_from_wildcard_path(
                            p, repo_files
                        )
                    ]
                elif isinstance(v, str):
                    new_mapping[k] = self._get_file_list_from_wildcard_path(
                        v, repo_files
                    )
                else:
                    raise NotImplementedError(
                        f"Loader does not support input 'data_files' of type Mapping[{type(v)}]"
                    )

            self.data_files = new_mapping
        elif isinstance(self.data_files, list):
            assert all(isinstance(s, str) for s in self.data_files)
            self.data_files = [
                file
                for p in self.data_files
                for file in self._get_file_list_from_wildcard_path(p, repo_files)
            ]
        else:
            raise NotImplementedError(
                f"Loader does not support input 'data_files' of type {type(self.data_files)}"
            )

    def load_data(self):
        self.sef_default_data_classification(
            ["public"], "when loading from Huggingface spaces"
        )
        self._map_wildcard_path_to_full_paths()
        self.path = self._download_data()
        return super().load_data()
