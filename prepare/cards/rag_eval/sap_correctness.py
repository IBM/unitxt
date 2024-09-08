import sys
from ast import literal_eval

from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.collections_operators import Slice
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import (
    Apply,
    CastFields,
    Copy,
    ExecuteExpression,
    FilterByCondition,
    FilterByExpression,
    Set,
    Shuffle,
)
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card


def add_sap_correctness_card(filter_idk):
    first_preprocess_step = (
        [FilterByCondition(values={"is_idk_response": [0, "0"]}, condition="in")]
        if filter_idk
        else [FilterByCondition(values={"category": ["Deviant"]}, condition="not in")]
    )

    referenceless_card = TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="SAP",
            data_files={
                "test": "SAP_Human_Evaluations_jan_march_2024_per_example_with_idk_fixed.csv",
            },
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            *first_preprocess_step,
            FilterByCondition(values={"model_scoring_1": [None]}, condition="not in"),
            # FilterByExpression(expression="len(input_background) < 15000"),
            Slice(
                field_to_field={
                    "model_scoring_0": "model_scoring_0",
                    "model_scoring_1": "model_scoring_1",
                },
                stop=-1,
            ),
            CastFields(fields={"model_scoring_0": "int", "model_scoring_1": "int"}),
            FilterByExpression(
                expression="(model_scoring_0 >= 75 and model_scoring_1 >= 75) "
                "or (model_scoring_0 < 75 and model_scoring_1 < 75)"
            ),  # annotators agree
            ExecuteExpression(
                expression="model_scoring_0 >= 75 and model_scoring_1 >= 75",
                to_field="is_correct",
            ),
            Set(fields={"choices": ["no", "yes"]}),
            CastFields(fields={"is_correct": "int"}),
            Copy(
                field_to_field={
                    "is_correct": "number_val",
                    "input": "question",
                    "input_background": "contexts",
                    "model_prediction": "answer",
                }
            ),
            Apply("contexts", function=literal_eval, to_field="contexts"),
            MapInstanceValues(
                mappers={"is_correct": {str(0): ["no"], str(1): ["yes"]}},
                strict=False,
            ),
            Copy(field_to_field={"is_correct": "textual_label"}),
            Shuffle(page_size=sys.maxsize),
        ],
        task="tasks.rag_eval.correctness_holistic.binary",
        templates="templates.rag_eval.correctness_holistic.all",
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
    filter_idk_str = "_no_idk" if filter_idk else ""
    add_to_catalog(
        referenceless_card,
        f"cards.rag_eval.sap_correctness{filter_idk_str}",
    )


for filter_idk in [True, False]:
    add_sap_correctness_card(filter_idk)
