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

Operators in Unitxt catalog:
LoadHF : loads from Huggingface dataset.
LoadCSV: loads from csv (comma separated value) files
LoadFromKaggle: loads datasets from the kaggle.com community site
LoadFromIBMCloud: loads a dataset from the IBM cloud.
------------------------
"""
import itertools
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

from .dataclass import InternalField, OptionalField
from .fusion import FixedFusion
from .logging_utils import get_logger
from .operator import SourceOperator
from .settings_utils import get_settings
from .stream import MultiStream, Stream

logger = get_logger()
settings = get_settings()


class Loader(SourceOperator):
    # The loader_limit an optional parameter used to control the maximum number of instances to load from the the source.
    # It is usually provided to the loader via the recipe (see standard.py)
    # The loader can use this value to limit the amount of data downloaded from the source
    # to reduce loading time.  However, this may not always be possible, so the
    # loader may ignore this.  In any case, the recipe, will limit the number of instances in the returned
    # stream, after load is complete.
    loader_limit: int = None
    streaming: bool = False

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


class LoadHF(Loader):
    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    split: Optional[str] = None
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None
    streaming: bool = True
    filtering_lambda: Optional[str] = None
    _cache: dict = InternalField(default=None)
    requirements_list: List[str] = OptionalField(default_factory=list)

    def verify(self):
        for requirement in self.requirements_list:
            if requirement not in self._requirements_list:
                self._requirements_list.append(requirement)
        super().verify()

    def filtered_load(self, dataset):
        logger.info(f"\nLoading filtered by: {self.filtering_lambda};")
        return MultiStream(
            {
                name: dataset[name].filter(eval(self.filtering_lambda))
                for name in dataset
            }
        )

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
                    )
                except ValueError as e:
                    if "trust_remote_code" in str(e):
                        raise ValueError(
                            f"{self.__class__.__name__} cannot run remote code from huggingface without setting unitxt.settings.allow_unverified_code=True or by setting environment variable: UNITXT_ALLOW_UNVERIFIED_CODE."
                        ) from e

            if self.filtering_lambda is not None:
                dataset = self.filtered_load(dataset)

            if self.split is not None:
                dataset = {self.split: dataset}

            self._cache = dataset
        else:
            dataset = self._cache

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
                    )
                except ValueError as e:
                    if "trust_remote_code" in str(e):
                        raise ValueError(
                            f"{self.__class__.__name__} cannot run remote code from huggingface without setting unitxt.settings.allow_unverified_code=True or by setting environment variable: UNITXT_ALLOW_UNVERIFIED_CODE."
                        ) from e

            if self.filtering_lambda is not None:
                dataset = self.filtered_load(dataset)

            if self.split is None:
                for split in dataset.keys():
                    dataset[split] = dataset[split].to_iterable_dataset()
            else:
                dataset = {self.split: dataset}

            self._cache = dataset
        else:
            dataset = self._cache

        return dataset

    def split_limited_load(self, split_name):
        yield from itertools.islice(self._cache[split_name], self.get_limit())

    def limited_load(self):
        self.log_limited_loading()
        return MultiStream(
            {
                name: Stream(
                    generator=self.split_limited_load, gen_kwargs={"split_name": name}
                )
                for name in self._cache.keys()
            }
        )

    def process(self):
        try:
            dataset = self.stream_dataset()
        except (
            NotImplementedError
        ):  # streaming is not supported for zipped files so we load without streaming
            dataset = self.load_dataset()

        if self.get_limit() is not None:
            return self.limited_load()

        return MultiStream.from_iterables(dataset)


class LoadCSV(Loader):
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

    def process(self):
        if self.streaming:
            return MultiStream(
                {
                    name: Stream(generator=self.stream_csv, gen_kwargs={"file": file})
                    for name, file in self.files.items()
                }
            )

        return MultiStream(
            {
                name: Stream(generator=self.load_csv, gen_kwargs={"file": file})
                for name, file in self.files.items()
            }
        )


class LoadFromSklearn(Loader):
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

    def process(self):
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
    url: str
    _requirements_list: List[str] = ["opendatasets"]

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

    def process(self):
        with TemporaryDirectory() as temp_directory:
            self.downloader(self.url, temp_directory)
            dataset = hf_load_dataset(temp_directory, streaming=False)

        return MultiStream.from_iterables(dataset)


class LoadFromIBMCloud(Loader):
    endpoint_url_env: str
    aws_access_key_id_env: str
    aws_secret_access_key_env: str
    bucket_name: str
    data_dir: str = None

    # Can be either:
    # 1. a list of file names, the split of each file is determined by the file name pattern
    # 2. Mapping: split -> file_name, e.g. {"test" : "test.json", "train": "train.json"}
    # 3. Mapping: split -> file_names, e.g. {"test" : ["test1.json", "test2.json"], "train": ["train.json"]}
    data_files: Union[Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    caching: bool = True
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

    def process(self):
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
                    os.rename(
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
    """Allow loading data from multiple sources.

    Examples:
    1) Loading the train split from Huggingface hub and the test set from a local file:

    MultipleSourceLoader(loaders = [ LoadHF(path="public/data",split="train"), LoadCSV({"test": "mytest.csv"}) ])

    2) Loading a test set combined from two files

    MultipleSourceLoader(loaders = [ LoadCSV({"test": "mytest1.csv"}, LoadCSV({"test": "mytest2.csv"}) ])


    """

    sources: List[Loader]

    def process(self):
        return FixedFusion(
            origins=self.sources, max_instances_per_origin=self.get_limit()
        ).process()
