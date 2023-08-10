import os
from tempfile import TemporaryDirectory
from typing import Mapping, Optional, Sequence, Union

from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

from .operator import SourceOperator
from .stream import MultiStream

try:
    import ibm_boto3
    from ibm_botocore.client import ClientError

    ibm_boto3_available = True
except ImportError:
    ibm_boto3_available = False


class Loader(SourceOperator):
    pass


class LoadHF(Loader):
    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None
    streaming: bool = True
    cached = False

    def process(self):
        dataset = hf_load_dataset(
            self.path, name=self.name, data_dir=self.data_dir, data_files=self.data_files, streaming=self.streaming
        )

        return MultiStream.from_iterables(dataset)


class LoadFromIBMCloud(Loader):
    endpoint_url_env: str
    aws_access_key_id_env: str
    aws_secret_access_key_env: str
    bucket_name: str
    data_dir: str
    data_files: Sequence[str]

    def _download_from_cos(self, cos, bucket_name, item_name, local_file):
        print(f"Downloading {item_name} from {bucket_name} COS to {local_file}")
        try:
            response = cos.Object(bucket_name, item_name).get()
            size = response["ContentLength"]
        except Exception as e:
            raise Exception(f"Unabled to access {item_name} in {bucket_name} in COS", e)

        progress_bar = tqdm(total=size, unit="iB", unit_scale=True)

        def upload_progress(chunk):
            progress_bar.update(chunk)

        try:
            cos.Bucket(bucket_name).download_file(item_name, local_file, Callback=upload_progress)
            print("\nDownload Successful")
        except Exception as e:
            raise Exception(f"Unabled to download {item_name} in {bucket_name}", e)

    def prepare(self):
        super().prepare()
        self.endpoint_url = os.getenv(self.endpoint_url_env)
        self.aws_access_key_id = os.getenv(self.aws_access_key_id_env)
        self.aws_secret_access_key = os.getenv(self.aws_secret_access_key_env)

    def verify(self):
        super().verify()
        assert (
            ibm_boto3_available
        ), f"Please install ibm_boto3 in order to use the LoadFromIBMCloud loader (using `pip install ibm-cos-sdk`) "
        assert self.endpoint_url is not None, f"Please set the {self.endpoint_url_env} environmental variable"
        assert self.aws_access_key_id is not None, f"Please set {self.aws_access_key_id_env} environmental variable"
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

        with TemporaryDirectory() as temp_directory:
            for data_file in self.data_files:
                self._download_from_cos(
                    cos, self.bucket_name, self.data_dir + "/" + data_file, temp_directory + "/" + data_file
                )
            dataset = hf_load_dataset(temp_directory, streaming=False)

        return MultiStream.from_iterables(dataset)
