import json
import os
import tempfile
from unittest.mock import patch

import ibm_boto3
import pandas as pd
from unitxt.loaders import (
    LoadCSV,
    LoadFromDictionary,
    LoadFromIBMCloud,
    LoadHF,
    MultipleSourceLoader,
)
from unitxt.logging_utils import get_logger

from tests.utils import UnitxtTestCase

logger = get_logger()


CONTENT = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


class DummyBody:
    def iter_lines():
        for line in CONTENT:
            line_as_str = json.dumps(line)
            line_as_bytes = line_as_str.encode()
            yield line_as_bytes


class DummyObject:
    def get(self):
        return {"ContentLength": 1, "Body": DummyBody}


class DummyBucket:
    def download_file(self, item_name, local_file, Callback):
        with open(local_file, "w") as f:
            logger.info(local_file)
            for line in CONTENT:
                f.write(json.dumps(line) + "\n")


class DummyS3:
    def Object(self, bucket_name, item_name):
        return DummyObject()

    def Bucket(self, bucket_name):
        return DummyBucket()


class TestLoaders(UnitxtTestCase):
    def test_load_csv(self):
        # Using a context for the temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = {}
            dfs = {}

            for file in ["train", "test"]:
                path = os.path.join(tmp_dir, file + ".csv")  # Adding a file extension
                df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})  # Replace with your data
                dfs[file] = df
                df.to_csv(path, index=False)
                files[file] = path

            loader = LoadCSV(files=files)
            ms = loader()

            for file in ["train", "test"]:
                for saved_instance, loaded_instance in zip(
                    dfs[file].iterrows(), ms[file]
                ):
                    self.assertEqual(saved_instance[1].to_dict(), loaded_instance)

    def test_load_csv_with_pandas_args(self):
        # Using a context for the temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = {}
            dfs = {}

            for file in ["train", "test"]:
                path = os.path.join(tmp_dir, file + ".tsv")  # Adding a file extension
                df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})  # Replace with your data
                dfs[file] = df
                df.to_csv(path, index=False, sep="\t")
                files[file] = path

            loader = LoadCSV(files=files, sep="\t")
            ms = loader()

            for file in ["train", "test"]:
                for saved_instance, loaded_instance in zip(
                    dfs[file].iterrows(), ms[file]
                ):
                    self.assertEqual(saved_instance[1].to_dict(), loaded_instance)

    def test_load_from_ibm_cos(self):
        os.environ["DUMMY_URL_ENV"] = "DUMMY_URL"
        os.environ["DUMMY_KEY_ENV"] = "DUMMY_KEY"
        os.environ["DUMMY_SECRET_ENV"] = "DUMMY_SECRET"
        for data_files in [
            ["train.jsonl", "test.jsonl"],
            {"train": "train.jsonl", "test": "test.jsonl"},
            {"train": ["train.jsonl"], "test": ["test.jsonl"]},
        ]:
            for loader_limit in [1, 2, None]:
                loader = LoadFromIBMCloud(
                    endpoint_url_env="DUMMY_URL_ENV",
                    aws_access_key_id_env="DUMMY_KEY_ENV",
                    aws_secret_access_key_env="DUMMY_SECRET_ENV",
                    bucket_name="DUMMY_BUCKET",
                    data_dir="DUMMY_DATA_DIR",
                    data_files=data_files,
                    loader_limit=loader_limit,
                )
                with patch.object(ibm_boto3, "resource", return_value=DummyS3()):
                    ms = loader()
                    ds = ms.to_dataset()
                    if loader_limit is None:
                        self.assertEqual(len(ds["test"]), 2)
                    else:
                        self.assertEqual(len(ds["test"]), loader_limit)
                    self.assertEqual(ds["test"][0], {"a": 1, "b": 2})

    def test_load_from_HF_compressed(self):
        loader = LoadHF(path="GEM/xlsum", name="igbo")  # the smallest file
        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            ms.to_dataset()["train"][0]["url"],
            "https://www.bbc.com/igbo/afirika-43986554",
        )
        assert set(dataset.keys()) == {
            "train",
            "validation",
            "test",
        }, f"Unexpected fold {dataset.keys()}"

    def test_load_from_HF_compressed_split(self):
        loader = LoadHF(
            path="GEM/xlsum", name="igbo", split="train"
        )  # the smallest file
        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            ms.to_dataset()["train"][0]["url"],
            "https://www.bbc.com/igbo/afirika-43986554",
        )
        assert list(dataset.keys()) == ["train"], f"Unexpected fold {dataset.keys()}"

    def test_load_from_HF(self):
        loader = LoadHF(path="sst2")
        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            dataset["train"][0]["sentence"],
            "hide new secretions from the parental units ",
        )
        assert set(dataset.keys()) == {
            "train",
            "validation",
            "test",
        }, f"Unexpected fold {dataset.keys()}"

    def test_load_from_HF_split(self):
        loader = LoadHF(path="sst2", split="train")
        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            dataset["train"][0]["sentence"],
            "hide new secretions from the parental units ",
        )
        assert list(dataset.keys()) == ["train"], f"Unexpected fold {dataset.keys()}"

    def test_load_from_HF_filter(self):
        loader = LoadHF(
            path="CohereForAI/aya_evaluation_suite",
            name="aya_human_annotated",
            filtering_lambda='lambda instance: instance["language"]=="eng"',
        )
        ms = loader.stream_dataset()
        dataset = ms.to_dataset()
        self.assertEqual(
            list(dataset.keys()), ["test"]
        )  # that HF dataset only has the 'test' split
        self.assertEqual(dataset["test"][0]["language"], "eng")
        ms = loader.load_dataset()
        dataset = ms.to_dataset()
        self.assertEqual(
            list(dataset.keys()), ["test"]
        )  # that HF dataset only has the 'test' split
        self.assertEqual(dataset["test"][0]["language"], "eng")

    def test_multiple_source_loader(self):
        # Using a context for the temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = {}
            dfs = {}

            for file in ["train", "test"]:
                path = os.path.join(tmp_dir, file + ".csv")  # Adding a file extension
                if file == "train":
                    df = pd.DataFrame({"x": ["train_1", "train_2"]})
                else:
                    df = pd.DataFrame({"x": ["test_1", "test_2", "test_3"]})
                dfs[file] = df
                df.to_csv(path, index=False)
                files[file] = path

            loader = MultipleSourceLoader(
                sources={
                    "loadCSV_train": LoadCSV(files={"train": files["train"]}),
                    "loadCSV_test": LoadCSV(files={"test": files["test"]}),
                }
            )
            ms = loader()

            for file in ["train", "test"]:
                assert len(dfs[file]) == len(list(ms[file]))
                for saved_instance, loaded_instance in zip(
                    dfs[file].iterrows(), ms[file]
                ):
                    loaded_instance.pop("group")
                    self.assertEqual(saved_instance[1].to_dict(), loaded_instance)

            loader = MultipleSourceLoader(
                sources={
                    "loadCSV_train": LoadCSV(files={"train": files["train"]}),
                    "loadCSV_test": LoadCSV(files={"test": files["test"]}),
                }
            )
            ms = loader()
            assert len(dfs["test"]) == len(list(ms["test"]))
            assert len(dfs["train"]) == len(list(ms["train"]))

    def test_load_from_dictionary(self):
        data = {
            "train": [
                {"input": "Input1", "output": "Result1"},
                {"input": "Input2", "output": "Result2"},
            ],
            "test": [
                {"input": "Input3", "output": "Result3"},
            ],
        }
        loader = LoadFromDictionary(data=data)
        streams = loader.process()

        for split, instances in data.items():
            for original_instance, stream_instance in zip(instances, streams[split]):
                self.assertEqual(original_instance, stream_instance)
