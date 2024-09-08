import sys
from ast import literal_eval

import numpy as np
from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import (
    Apply,
    CastFields,
    Copy,
    ExecuteExpression,
    FilterByCondition,
    ListFieldValues,
    Set,
    Shuffle,
)
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card


def add_tls_correctness_card(partial_policy, task_type):
    first_preprocess_step = (
        [FilterByCondition(values={"label": [2, "2"]}, condition="not in")]
        if partial_policy == "_no_partial"
        else []
    )
    min_pos_response = "3" if partial_policy == "_partial_neg" else "2"
    gt_processing_steps = (
        [
            MapInstanceValues(
                mappers={"gold_answer": {str(np.nan): "", str(None): ""}}, strict=False
            ),
            ListFieldValues(fields=["gold_answer"], to_field="ground_truths"),
        ]
        if task_type == "answer_correctness"
        else []
    )

    # noinspection PyTypeChecker
    referenceless_card = TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="tls",
            data_files={
                "test": "specv-san-annotated.csv",
            },
            data_classification_policy=["public"],
        ),
        preprocess_steps=first_preprocess_step
        + gt_processing_steps
        + [
            FilterByCondition(values={"gold_answer": None}, condition="ne"),
            CastFields(fields={"label": "int"}),
            ExecuteExpression(
                expression=f"label >= {min_pos_response}",
                to_field="is_correct",
            ),
            Set(fields={"choices": ["no", "yes"]}),
            CastFields(fields={"is_correct": "int"}),
            Apply("contexts", function=literal_eval, to_field="contexts"),
            Copy(
                field_to_field={
                    "is_correct": "number_val",
                    "response": "answer",
                }
            ),
            MapInstanceValues(
                mappers={"is_correct": {str(0): ["no"], str(1): ["yes"]}},
                strict=False,
            ),
            Copy(field_to_field={"is_correct": "textual_label"}),
            Shuffle(page_size=sys.maxsize),
        ],
        task=f"tasks.rag_eval.{task_type}.binary",
        templates=f"templates.rag_eval.{task_type}.all",
        sampler=DiverseLabelsSampler(
            choices="choices", labels="textual_label", include_empty_label=False
        ),
    )

    test_card(
        referenceless_card,
        num_demos=2,
        strict=False,
        demos_removed_from_data=False,
        demos_taken_from="test",
        demos_pool_size=20,
    )
    task_type_in_card = task_type.replace("_holistic", "")
    add_to_catalog(
        referenceless_card,
        f"cards.rag_eval.tls_{task_type_in_card}{partial_policy}",
    )


for partial_policy in ["", "_no_partial", "_partial_neg"]:
    for task_type in ["correctness_holistic", "answer_correctness"]:
        add_tls_correctness_card(partial_policy, task_type)
