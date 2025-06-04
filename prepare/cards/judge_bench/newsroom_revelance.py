from typing import Any

from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadJsonFile
from unitxt.operators import Cast, Rename, Set
from unitxt.task import Task
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadJsonFile(
        files={
            "test": "https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/newsroom/newsroom.json",
        },
        data_classification_policy=["public"],
        data_field="instances",
    ),
    preprocess_steps=[
        Rename(field="instance", to_field="context"),
        Rename(field="annotations/Relevance/mean_human", to_field="mean_score"),
        Cast(field="mean_score", to="float"),
        Set(
            fields={
                "criteria": "metrics.llm_as_judge.direct.criteria.summarization_relevance"
            }
        ),
    ],
    task=Task(
        input_fields={"context": str, "criteria": Any},
        reference_fields={"mean_score": float},
        prediction_type=float,
        metrics=[
            "metrics.spearman",
        ],
        default_template="templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]",
    ),
    templates=[
        "templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]"
    ],
)

test_card(card, demos_taken_from="test", strict=False)

add_to_catalog(
    card,
    "cards.judge_bench.newswoom.relevance",
    overwrite=True,
)
