from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import (
    CastFields,
    Copy,
    ExecuteExpression,
    FilterByCondition,
    ListFieldValues,
    Set,
)
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card

type_to_field_names = {
    "faithfulness": (
        "faithfulness_scores",
        "is_faithful",
    ),
    "correctness_holistic": (
        "appropriateness_score",
        "is_correct",
    ),
    "answer_correctness": (
        "appropriateness_score",
        "is_correct",
    ),
}

for metric_name, (
    score_field,
    agg_field_name,
) in type_to_field_names.items():
    for subset in ["", "_no_idk"]:
        preprocessing_steps = []
        if subset == "_no_idk":
            preprocessing_steps.append(
                FilterByCondition(values={"is_idk": [0, "0"]}, condition="in")
            )
        if metric_name == "answer_correctness":
            preprocessing_steps.append(
                ListFieldValues(fields=["gold_answer"], to_field="ground_truths")
            )

        card = TaskCard(
            loader=LoadFromIBMCloud(
                endpoint_url_env="FMEVAL_COS_URL",
                aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
                aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
                bucket_name="metrics-eval-data",
                data_dir="RAGEval",
                data_files={
                    "test": "rageval_annotated_data_with_idk.csv",
                },
                data_classification_policy=["public"],
            ),
            preprocess_steps=preprocessing_steps[
                *preprocessing_steps,
                Set(fields={"choices": ["no", "yes"]}),
                ExecuteExpression(
                    expression=f"{score_field} >= 3",
                    to_field=agg_field_name,
                ),
                CastFields(fields={agg_field_name: "int"}),
                Copy(
                    field_to_field={
                        agg_field_name: "number_val",
                        "query": "question",
                        "ret_passages_text": "contexts",
                        "gen_answer": "answer",
                    }
                ),
                ListFieldValues(fields=["contexts"], to_field="contexts"),
                MapInstanceValues(
                    mappers={agg_field_name: {str(0): ["no"], str(1): ["yes"]}},
                    strict=False,
                ),
                Copy(field_to_field={agg_field_name: "textual_label"}),
            ],
            task=f"tasks.rag_eval.{metric_name}.binary",
            templates=f"templates.rag_eval.{metric_name}.all",
            sampler=DiverseLabelsSampler(
                choices="choices", labels="textual_label", include_empty_label=False
            ),
        )

        test_card(
            card,
            num_demos=2,
            strict=False,
            demos_removed_from_data=False,
            demos_taken_from="test",
            demos_pool_size=20,
        )
        add_to_catalog(
            card, f"cards.rag_eval.rageval{subset}_{metric_name}", overwrite=True
        )
