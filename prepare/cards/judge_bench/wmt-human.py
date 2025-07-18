from typing import Any

from unitxt.blocks import (
    Rename,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadJsonFile
from unitxt.operators import Cast, ExecuteExpression, Set
from unitxt.task import Task
from unitxt.test_utils.card import test_card

dataset_to_config = {
    "en_de": {"source_language": "english", "target_language": "german", "url": ""},
    "zh_en": {
        "source_language": "chinese",
        "target_language": "english",
    },
}

for dataset_name, config in dataset_to_config.items():
    card = TaskCard(
        loader=LoadJsonFile(
            files={
                "test": f"https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/wmt-human/wmt-human_{dataset_name}.json",
            },
            data_classification_policy=["public"],
            data_field="instances",
        ),
        preprocess_steps=[
            Rename(field="annotations/quality/mean_human", to_field="mean_score"),
            Cast(field="mean_score", to="float"),
            ExecuteExpression(expression="mean_score/6", to_field="mean_score"),
            Rename(
                field_to_field={
                    "instance/source": "source text",
                    "instance/reference": "reference translation",
                    "instance/translation": "translation",
                }
            ),
            Set(
                fields={
                    "criteria": "metrics.llm_as_judge.direct.criteria.translation_quality",
                    "source language": config["source_language"],
                    "target language": config["target_language"],
                }
            ),
        ],
        task=Task(
            input_fields={
                "source text": str,
                "source language": str,
                "target language": str,
                "reference translation": str,
                "translation": str,
                "criteria": Any,
            },
            reference_fields={"mean_score": float},
            prediction_type=float,
            metrics=["metrics.spearman"],
            default_template="templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]",
        ),
        templates=[
            "templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False)

    add_to_catalog(
        card,
        f"cards.judge_bench.wmt_human.{config['source_language']}_to_{config['target_language']}.quality",
        overwrite=True,
    )
