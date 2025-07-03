import re
from typing import Any

from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadJsonFile
from unitxt.operators import Cast, Rename, Set
from unitxt.processors import GroupDictWithRegex
from unitxt.task import Task
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadJsonFile(
        files={
            "test": "https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/roscoe/roscoe-cosmos-overall.json",
        },
        data_classification_policy=["public"],
        data_field="instances",
    ),
    preprocess_steps=[
        Rename(field="annotations/Coherency/mean_human", to_field="mean_score"),
        Cast(field="mean_score", to="float"),
        GroupDictWithRegex(
            field="instance",
            pattern=(
                r".*?Situation \(Premise\):\s+"
                r"(?P<premise>.*?)\s+"
                r"Claim \(Hypothesis\):\s+"
                r"(?P<hypothesis>.*?)\s+"
                r"Is the Claim supported by the Situation\?\s+Correct Relationship \(Yes or No\):\s"
                r"(?P<correct_answer>.*?)\s+"
                r"GENERATED RESPONSE:\s+"
                r"(?P<generated_response>.*?)\s+"
                r"Judge the generated response:"
            ),
            flags=re.DOTALL,
        ),
        Rename(
            field_to_field={
                "instance/premise": "premise",
                "instance/hypothesis": "hypothesis",
                "instance/generated_response": "generated response",
                "instance/correct_answer": "correct answer",
            }
        ),
        Set(
            fields={
                "criteria": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_coherency",
                "question": "Is the Hypothesis supported by the Premise?",
            }
        ),
    ],
    task=Task(
        input_fields={
            "premise": str,
            "hypothesis": str,
            "question": str,
            "generated response": str,
            "correct answer": str,
            "criteria": Any,
        },
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
    "cards.judge_bench.roscoe.cosmos.overall.coherence",
    overwrite=True,
)
