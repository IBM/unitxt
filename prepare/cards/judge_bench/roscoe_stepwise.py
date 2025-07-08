import re
from typing import Any

from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadJsonFile
from unitxt.operators import Copy, MapInstanceValues, Rename, Set
from unitxt.processors import GroupDictWithRegex
from unitxt.task import Task
from unitxt.test_utils.card import test_card

roscoe_datasets = [
    "cosmos",
    "drop",
    "esnli",
    # "gsm8k" omitting because it is different
]

criteria_to_artifact = {
    "Grammar": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_bad_grammar",
    "Factuality": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_non_factual",
    "Coherency and Logic": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_non_coherent",
    "Final Answer": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_bad_final_answer",
    "Hallucination": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_hallucination",
    "Redundancy": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_redundancy",
    "Repetition": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_repetition",
    "Commonsense": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_commonsense",
    "Arithmetic": "metrics.llm_as_judge.direct.criteria.step_by_step_reasoning_arithmetic",
}

for criteria_name, criteria_artifact in criteria_to_artifact.items():
    for roscoe_dataset in roscoe_datasets:
        card = TaskCard(
            loader=LoadJsonFile(
                files={
                    "test": f"https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/roscoe/roscoe-{roscoe_dataset}-stepwise.json",
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
                        r"JUDGE:\s+"
                        r"(?P<judged_step>.*)\s+"
                    ),
                    flags=re.DOTALL,
                ),
                Rename(
                    field_to_field={
                        "instance/premise": "premise",
                        "instance/hypothesis": "hypothesis",
                        "instance/model_reasoning": "model reasoning",
                        "instance/correct_answer": "correct answer",
                        "instance/judged_step": "step",
                        f"annotations/{criteria_name}/majority_human": "label",
                    }
                ),
                MapInstanceValues(
                    mappers={
                        "label": {"no": "No", "yes": "Yes"},
                    }
                ),
                Copy(field="label", to_field="label_value"),
                MapInstanceValues(
                    mappers={
                        "label_value": {"Yes": 0.0, "No": 1.0},
                    }
                ),
                Set(
                    fields={
                        "criteria": criteria_artifact,
                        "question": "Is the Hypothesis supported by the Premise?",
                    }
                ),
            ],
            task=Task(
                input_fields={
                    "premise": str,
                    "hypothesis": str,
                    "question": str,
                    "model reasoning": str,
                    "correct answer": str,
                    "criteria": Any,
                    "step": str,
                },
                reference_fields={"label_value": float},
                prediction_type=float,
                metrics=["metrics.accuracy", "metrics.f1_macro"],
                default_template="templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]",
            ),
            templates=[],
        )

        test_card(card, demos_taken_from="test", strict=False)

        add_to_catalog(
            card,
            f"cards.judge_bench.roscoe.{roscoe_dataset}.stepwise.{criteria_name.lower().replace(' ', '_')}",
            overwrite=True,
        )
