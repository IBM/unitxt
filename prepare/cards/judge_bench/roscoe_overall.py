import re
from typing import Any

from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.llm_as_judge_constants import DirectCriteriaCatalogEnum
from unitxt.loaders import LoadJsonFile
from unitxt.operators import (
    Cast,
    Copy,
    ExecuteExpression,
    MapInstanceValues,
    Rename,
    Set,
)
from unitxt.processors import GroupDictWithRegex
from unitxt.task import Task
from unitxt.test_utils.card import test_card

roscoe_datasets = [
    "cosmos",
    "drop",
    "esnli",
    # "gsm8k" omitting because it is different
]

criteria_to_config = {
    "coherence": {
        "label_mapping": {"annotations/Coherency/mean_human": "mean_score"},
        "criteria_artifact": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_coherency",
        "preprocess_steps": [
            Cast(field="mean_score", to="float"),
            ExecuteExpression(expression="(mean_score - 1) / 4", to_field="mean_score"),
        ],
        "reference_fields": {"mean_score": float},
        "metrics": ["metrics.pearson", "metrics.spearman"],
    },
    "contradiction": {
        "label_mapping": {"annotations/Contradiction/majority_human": "label"},
        "criteria_artifact": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_contradiction",
        "preprocess_steps": [
            MapInstanceValues(
                mappers={
                    "label": {"no": "No", "yes": "Yes"},
                }
            ),
            Copy(field="label", to_field="label_value"),
            MapInstanceValues(
                mappers={
                    "label_value": DirectCriteriaCatalogEnum.STEP_BY_STEP_REASONING_MISSING_STEPS.value.option_map,
                }
            ),
        ],
        "reference_fields": {"label_value": float},
        "metrics": ["metrics.accuracy", "metrics.f1_macro"],
    },
    "missing_steps": {
        "label_mapping": {"annotations/Missing Steps/majority_human": "label"},
        "criteria_artifact": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_missing_steps",
        "preprocess_steps": [
            MapInstanceValues(
                mappers={
                    "label": {"no": "No", "yes": "Yes"},
                }
            ),
            Copy(field="label", to_field="label_value"),
            MapInstanceValues(
                mappers={
                    "label_value": DirectCriteriaCatalogEnum.STEP_BY_STEP_REASONING_MISSING_STEPS.value.option_map,
                }
            ),
        ],
        "reference_fields": {"label_value": float},
        "metrics": ["metrics.accuracy", "metrics.f1_macro"],
    },
    "overall_quality": {
        "label_mapping": {"annotations/Overall Quality/mean_human": "mean_score"},
        "criteria_artifact": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_overall_quality",
        "preprocess_steps": [
            Cast(field="mean_score", to="float"),
            ExecuteExpression(expression="(mean_score - 1) / 4", to_field="mean_score"),
        ],
        "reference_fields": {"mean_score": float},
        "metrics": ["metrics.pearson", "metrics.spearman"],
    },
}
for criteria_name, config in criteria_to_config.items():
    for roscoe_dataset in roscoe_datasets:
        card = TaskCard(
            loader=LoadJsonFile(
                files={
                    "test": f"https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/roscoe/roscoe-{roscoe_dataset}-overall.json",
                },
                data_classification_policy=["public"],
                data_field="instances",
            ),
            preprocess_steps=[
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
                        r"(?P<model_reasoning>.*?)\s+"
                        r"Judge the generated response:"
                    ),
                    flags=re.DOTALL,
                ),
                Rename(
                    field_to_field={
                        "instance/premise": "premise",
                        "instance/hypothesis": "hypothesis",
                        "instance/model_reasoning": "generated response",
                        "instance/correct_answer": "correct answer",
                        **config["label_mapping"],
                    }
                ),
                *config["preprocess_steps"],
                Set(
                    fields={
                        "criteria": config["criteria_artifact"],
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
                reference_fields=config["reference_fields"],
                prediction_type=float,
                metrics=config["metrics"],
                default_template="templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]",
            ),
            templates=[],
        )

        test_card(card, demos_taken_from="test", strict=False)

        add_to_catalog(
            card,
            f"cards.judge_bench.roscoe.overall.{roscoe_dataset}.{criteria_name}",
            overwrite=True,
        )
