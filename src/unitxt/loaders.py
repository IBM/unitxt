import itertools
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import pandas as pd
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

from .logging import get_logger
from .operator import SourceOperator
from .stream import MultiStream, Stream

logger = get_logger()
try:
    import ibm_boto3
    # from ibm_botocore.client import ClientError

    ibm_boto3_available = True
except ImportError:
    ibm_boto3_available = False


class Loader(SourceOperator):
    # The loader_limit an optional parameter used to control the maximum number of instances to load from the the source.
    # It is usually provided to the loader via the recipe (see standard.py)
    # The loader can use this value to limit the amount of data downloaded from the source
    # to reduce loading time.  However, this may not always be possible, so the
    # loader may ingore this.  In any case, the recipe, will limit the number of instances in the returned
    # stream after, after load is complete.
    loader_limit: int = None
    pass


class LoadHF(Loader):
    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    split: Optional[str] = None
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
    ] = None
    streaming: bool = True

    def process(self):
        try:
            with tempfile.TemporaryDirectory() as dir_to_be_deleted:
                dataset = hf_load_dataset(
                    self.path,
                    name=self.name,
                    data_dir=self.data_dir,
                    data_files=self.data_files,
                    streaming=self.streaming,
                    cache_dir=None if self.streaming else dir_to_be_deleted,
                    split=self.split,
                )
            if self.split is not None:
                dataset = {self.split: dataset}
        except (
            NotImplementedError
        ):  # streaming is not supported for zipped files so we load without streaming
            with tempfile.TemporaryDirectory() as dir_to_be_deleted:
                dataset = hf_load_dataset(
                    self.path,
                    name=self.name,
                    data_dir=self.data_dir,
                    data_files=self.data_files,
                    streaming=False,
                    keep_in_memory=True,
                    cache_dir=dir_to_be_deleted,
                    split=self.split,
                )
            if self.split is None:
                for split in dataset.keys():
                    dataset[split] = dataset[split].to_iterable_dataset()
            else:
                dataset = {self.split: dataset}

        return MultiStream.from_iterables(dataset)


class LoadCSV(Loader):
    files: Dict[str, str]
    chunksize: int = 1000

    def load_csv(self, file):
        for chunk in pd.read_csv(file, chunksize=self.chunksize):
            for _index, row in chunk.iterrows():
                yield row.to_dict()

    def process(self):
        return MultiStream(
            {
                name: Stream(generator=self.load_csv, gen_kwargs={"file": file})
                for name, file in self.files.items()
            }
        )


class LoadFromIBMCloud(Loader):
    endpoint_url_env: str
    aws_access_key_id_env: str
    aws_secret_access_key_env: str
    bucket_name: str
    data_dir: str = None
    data_files: Sequence[str]
    caching: bool = True

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

        if self.loader_limit is not None:
            if item_name.endswith(".jsonl"):
                first_lines = list(
                    itertools.islice(body.iter_lines(), self.loader_limit)
                )
                with open(local_file, "wb") as downloaded_file:
                    for line in first_lines:
                        downloaded_file.write(line)
                        downloaded_file.write(b"\n")
                logger.info(
                    f"\nDownload successful limited to {self.loader_limit} lines"
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
        assert ibm_boto3_available, "Please install ibm_boto3 in order to use the LoadFromIBMCloud loader (using `pip install ibm-cos-sdk`) "
        assert (
            self.endpoint_url is not None
        ), f"Please set the {self.endpoint_url_env} environmental variable"
        assert (
            self.aws_access_key_id is not None
        ), f"Please set {self.aws_access_key_id_env} environmental variable"
        assert (
            self.aws_secret_access_key is not None
        ), f"Please set {self.aws_secret_access_key_env} environmental variable"

    def process(self):
        cos = ibm_boto3.resource(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
        )
        local_dir = os.path.join(self.cache_dir, self.bucket_name, self.data_dir)
        if not os.path.exists(local_dir):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
        for data_file in self.data_files:
            local_file = os.path.join(local_dir, data_file)
            if not self.caching or not os.path.exists(local_file):
                # Build object key based on parameters. Slash character is not
                # allowed to be part of object key in IBM COS.
                object_key = (
                    self.data_dir + "/" + data_file
                    if self.data_dir is not None
                    else data_file
                )
                self._download_from_cos(
                    cos, self.bucket_name, object_key, local_dir + "/" + data_file
                )
        dataset = hf_load_dataset(local_dir, streaming=False)

        return MultiStream.from_iterables(dataset)
