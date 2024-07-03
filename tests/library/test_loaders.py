import json
import os
import tempfile
from unittest.mock import patch

import ibm_boto3
import pandas as pd
from unitxt.loaders import (
    LoadCSV,
    LoadFromDictionary,
    LoadFromHFSpace,
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
                df.to_csv(path, index=False)
                df["data_classification_policy"] = [
                    ["proprietary"] for _ in range(len(df))
                ]
                dfs[file] = df
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
            data_classification = ["pii", "propriety"]

            for file in ["train", "test"]:
                path = os.path.join(tmp_dir, file + ".tsv")  # Adding a file extension
                df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})  # Replace with your data
                df.to_csv(path, index=False, sep="\t")
                df["data_classification_policy"] = [
                    data_classification for _ in range(len(df))
                ]
                dfs[file] = df
                files[file] = path

            loader = LoadCSV(
                files=files, sep="\t", data_classification_policy=data_classification
            )
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
                    data_classification_policy=["public"],
                )
                with patch.object(ibm_boto3, "resource", return_value=DummyS3()):
                    ms = loader()
                    ds = ms.to_dataset()
                    if loader_limit is None:
                        self.assertEqual(len(ds["test"]), 2)
                    else:
                        self.assertEqual(len(ds["test"]), loader_limit)
                    self.assertEqual(
                        ds["test"][0],
                        {"a": 1, "b": 2, "data_classification_policy": ["public"]},
                    )

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
        loader = LoadHF(path="sst2", loader_limit=10)
        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            dataset["train"][0]["sentence"],
            "hide new secretions from the parental units ",
        )
        self.assertEqual(
            dataset["train"][0]["data_classification_policy"],
            ["public"],
        )
        self.assertEqual(
            dataset["test"][0]["data_classification_policy"],
            ["public"],
        )
        assert set(dataset.keys()) == {
            "train",
            "validation",
            "test",
        }, f"Unexpected fold {dataset.keys()}"

    def test_load_from_HF_multiple_innvocation(self):
        loader = LoadHF(
            path="CohereForAI/aya_evaluation_suite",
            name="aya_human_annotated",
            # filtering_lambda='lambda instance: instance["language"]=="eng"',
        )
        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            list(dataset.keys()), ["test"]
        )  # that HF dataset only has the 'test' split
        self.assertEqual(dataset["test"][0]["language"], "arb")

        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            list(dataset.keys()), ["test"]
        )  # that HF dataset only has the 'test' split
        self.assertEqual(dataset["test"][0]["language"], "arb")

    def test_load_from_HF_multiple_innvocation_with_filter(self):
        loader = LoadHF(
            path="CohereForAI/aya_evaluation_suite",
            name="aya_human_annotated",
            filtering_lambda='lambda instance: instance["language"]=="eng"',
        )
        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            list(dataset.keys()), ["test"]
        )  # that HF dataset only has the 'test' split
        self.assertEqual(dataset["test"][0]["language"], "eng")

        ms = loader.process()
        dataset = ms.to_dataset()
        self.assertEqual(
            list(dataset.keys()), ["test"]
        )  # that HF dataset only has the 'test' split
        self.assertEqual(dataset["test"][0]["language"], "eng")

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
        ms = loader.process()
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
                df.to_csv(path, index=False)
                dfs[file] = df
                files[file] = path

            loader = MultipleSourceLoader(
                sources=[
                    LoadCSV(
                        files={"train": files["train"]},
                        data_classification_policy=["public"],
                    ),
                    LoadCSV(
                        files={"test": files["test"]},
                        data_classification_policy=["pii"],
                    ),
                ]
            )
            ms = loader()
            for file in ["train", "test"]:
                assert len(dfs[file]) == len(list(ms[file]))
                for saved_instance, loaded_instance in zip(
                    dfs[file].iterrows(), ms[file]
                ):
                    saved_instance_as_dict = saved_instance[1].to_dict()
                    if file == "train":
                        saved_instance_as_dict["data_classification_policy"] = [
                            "public"
                        ]
                    else:
                        saved_instance_as_dict["data_classification_policy"] = ["pii"]

                    self.assertEqual(saved_instance_as_dict, loaded_instance)

            loader = MultipleSourceLoader(
                sources=[
                    LoadCSV(files={"test": files["train"]}),
                    LoadCSV(files={"test": files["test"]}),
                ]
            )
            ms = loader()
            assert len(dfs["test"]) + len(dfs["train"]) == len(list(ms["test"]))

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
                original_instance["data_classification_policy"] = ["proprietary"]
                self.assertEqual(original_instance, stream_instance)

    def test_load_from_dictionary_errors(self):
        data = [
            {"input": "Input3", "output": "Result3"},
        ]

        with self.assertRaises(ValueError) as cm:
            LoadFromDictionary(data=data)
        self.assertEqual(
            str(cm.exception),
            f"Passed data to LoadFromDictionary is not of type Dict[str, List[Dict[str, Any]]].\n"
            f"Expected data should map between split name and list of instances.\n"
            f"Received value: {data}\n",
        )

        data = {
            "train": [
                {"input": "Input1", "output": "Result1"},
                {"input2": "Input2", "output": "Result2"},
            ],
        }
        with self.assertRaises(ValueError) as cm:
            LoadFromDictionary(data=data)
        self.assertEqual(
            str(cm.exception),
            f"Not all instances in split 'train' have the same fields.\n"
            f"instance {data['train'][1]} has different fields different from {data['train'][0]}",
        )

    def test_load_from_hf_space(self):
        params = {
            "space_name": "lmsys/mt-bench",
            "data_files": {
                "train": [
                    "data/mt_bench/model_answer/koala-13b.jsonl",
                    "data/mt_bench/model_answer/llama-13b.jsonl",
                ],
                "test": "data/mt_bench/model_answer/wizardlm-13b.jsonl",
            },
            "data_classification_policy": ["pii"],
        }

        expected_sample = {
            "question_id": 81,
            "model_id": "wizardlm-13b",
            "answer_id": "DKHvKJgtzsvHN2ZJ8a3o5C",
            "tstamp": 1686788249.913451,
            "data_classification_policy": ["pii"],
        }
        loader = LoadFromHFSpace(**params)
        ms = loader.process().to_dataset()
        actual_sample = ms["test"][0]
        actual_sample.pop("choices")
        self.assertEqual(expected_sample, actual_sample)

        params["loader_limit"] = 10
        loader = LoadFromHFSpace(**params)
        ms = loader.process().to_dataset()
        assert ms.shape["train"] == (10, 6) and ms.shape["test"] == (10, 6)
