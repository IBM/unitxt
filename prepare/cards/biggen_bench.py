from typing import Any

from unitxt import add_to_catalog
from unitxt.blocks import Copy, LoadHF, Set, Task, TaskCard
from unitxt.llm_as_judge import CreateCriteriaWithOptionsFromDict
from unitxt.operators import Cast, MergeStreams
from unitxt.test_utils.card import test_card

empty_criteria = {
    "name": "",
    "description": "",
    "options": [
        {"name": f"score{i+1}_description", "description": ""} for i in range(5)
    ],
    "prediction_field": "response",
    "context_fields": ["system_prompt", "input", "reference_answer"],
}

empty_criteria["option_map"] = {
    option["name"]: i / 4 for i, option in enumerate(empty_criteria["options"])
}

for is_multilingual in [False, True]:
    stream_name = f"{'multilingual_' if is_multilingual else ''}human_eval"
    card_name = f"cards.biggen_bench{'_multilingual' if is_multilingual else ''}.results.human_eval"
    card = TaskCard(
        loader=LoadHF(
            path="prometheus-eval/BiGGen-Bench-Results",
            splits=[stream_name],
        ),
        preprocess_steps=[
            MergeStreams(
                streams_to_merge=[stream_name],
                new_stream_name="test",
                add_origin_stream_name=True,
            ),
            Set(fields={"criteria": empty_criteria}),
            Cast(field="human_score", to="float"),
            Copy(
                field_to_field={
                    "task": "criteria/name",
                    "score_rubric/criteria": "criteria/description",
                    "score_rubric/score1_description": "criteria/options/0/description",
                    "score_rubric/score2_description": "criteria/options/1/description",
                    "score_rubric/score3_description": "criteria/options/2/description",
                    "score_rubric/score4_description": "criteria/options/3/description",
                    "score_rubric/score5_description": "criteria/options/4/description",
                }
            ),
            CreateCriteriaWithOptionsFromDict(field="criteria"),
        ],
        task=Task(
            input_fields={
                "system_prompt": str,
                "input": str,
                "response": str,
                "reference_answer": str,
                "criteria": Any,
            },
            reference_fields={"human_score": float},
            prediction_type=float,
            metrics=["metrics.spearman", "metrics.pearson"],
            default_template="templates.empty[postprocessors=[processors.cast_to_float_return_nan_if_failed]]",
        ),
        templates=[],
        __description__=(
            "BIGGEN-Bench (BiG Generation Benchmark) is a comprehensive evaluation benchmark designed to assess the capabilities of large language models (LLMs) across a wide range of tasks. This benchmark focuses on free-form text generation and employs fine-grained, instance-specific evaluation criteria. This card is aimed to be used to benchmark LLM judges using the human evaluations as the ground truth."
        ),
    )

    test_card(card, demos_taken_from="test", debug=False)
    add_to_catalog(artifact=card, name=card_name, overwrite=True)
