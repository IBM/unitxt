import os.path
import tempfile
import unittest
from pathlib import Path

from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    MapInstanceValues,
    RenderAutoFormatTemplate,
    SequentialRecipe,
    SplitRandomMix,
)
from src.unitxt.metrics import HuggingfaceMetric, MetricPipeline
from src.unitxt.operators import AddID, CastFields, CopyFields
from src.unitxt.stream import MultiStream, Stream
from src.unitxt.test_utils.environment import modified_environment
from src.unitxt.test_utils.storage import get_directory_size

wnli_recipe = SequentialRecipe(
    steps=[
        LoadHF(path="glue", name="wnli"),
        SplitRandomMix(
            mix={
                "train": "train[95%]",
                "validation": "train[5%]",
                "test": "validation",
            }
        ),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
                "instruction": "classify the relationship between the two sentences from the choices.",
                "dataset": "wnli",
            }
        ),
        FormTask(
            inputs=["choices", "instruction", "sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.accuracy"],
        ),
        RenderAutoFormatTemplate(),
    ]
)


rte_recipe = SequentialRecipe(
    steps=[
        LoadHF(path="glue", name="rte"),
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        AddFields(
            fields={"choices": ["entailment", "not entailment"], "dataset": "rte"}
        ),
        FormTask(
            inputs=["choices", "sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.accuracy"],
        ),
        RenderAutoFormatTemplate(),
    ]
)


squad_metric = MetricPipeline(
    main_score="f1",
    preprocess_steps=[
        AddID(),
        AddFields(
            {
                "prediction_template": {"prediction_text": "PRED", "id": "ID"},
                "reference_template": {
                    "answers": {"answer_start": [-1], "text": "REF"},
                    "id": "ID",
                },
            },
            use_deepcopy=True,
        ),
        CopyFields(
            field_to_field=[
                ["references", "reference_template/answers/text"],
                ["prediction", "prediction_template/prediction_text"],
                ["id", "prediction_template/id"],
                ["id", "reference_template/id"],
                ["reference_template", "references"],
                ["prediction_template", "prediction"],
            ],
            use_query=True,
        ),
    ],
    metric=HuggingfaceMetric(
        hf_metric_name="squad",
        main_score="f1",
        scale=100.0,
    ),
)

spearman_metric = MetricPipeline(
    main_score="spearmanr",
    preprocess_steps=[
        CopyFields(field_to_field=[("references/0", "references")], use_query=True),
        CastFields(
            fields={"prediction": "float", "references": "float"},
            failure_defaults={"prediction": 0.0},
            use_nested_query=True,
        ),
    ],
    metric=HuggingfaceMetric(
        hf_metric_name="spearmanr",
        main_score="spearmanr",
    ),
)

catalog_path = os.path.join(Path(__file__).parent, "temp_catalog")


class TestHfCache(unittest.TestCase):
    def test_caching_stream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with modified_environment(HF_DATASETS_CACHE=tmp_dir):
                self.assertEqual(get_directory_size(tmp_dir), 0)

                def gen():
                    for i in range(3):
                        yield {"x": i}

                ds = MultiStream({"test": Stream(generator=gen)}).to_dataset(
                    use_cache=True, cache_dir=tmp_dir
                )
                for i, item in enumerate(ds["test"]):
                    self.assertEqual(item["x"], i)
                self.assertNotEqual(get_directory_size(tmp_dir), 0)

    def test_caching_stream_with_general_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with modified_environment(HF_DATASETS_CACHE=tmp_dir):
                self.assertEqual(get_directory_size(tmp_dir), 0)

                def gen():
                    for i in range(3):
                        yield {"x": i}

                ds = MultiStream({"test": Stream(generator=gen)}).to_dataset(
                    use_cache=True
                )
                for i, item in enumerate(ds["test"]):
                    self.assertEqual(item["x"], i)
                self.assertNotEqual(get_directory_size(tmp_dir), 0)

    def test_not_caching_stream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with modified_environment(HF_DATASETS_CACHE=tmp_dir):
                self.assertEqual(get_directory_size(tmp_dir), 0)

                def gen():
                    for i in range(3):
                        yield {"x": i}

                ds = MultiStream({"test": Stream(generator=gen)}).to_dataset()
                for i, item in enumerate(ds["test"]):
                    self.assertEqual(item["x"], i)
                self.assertEqual(get_directory_size(tmp_dir), 0)


if __name__ == "__main__":
    unittest.main()
