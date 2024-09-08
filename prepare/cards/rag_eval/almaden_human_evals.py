from ast import literal_eval

from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import (
    Apply,
    CastFields,
    Copy,
    ExecuteExpression,
    FilterByCondition,
    FilterByExpression,
    ListFieldValues,
    Set,
)
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card

data_subsets = [
    # "longNQ",
    # "AF",
    # "askHR_SME",
    # "askETE",
    # "W3IT",
    # "Maximo",
    # "SamsungWellsFargo",
    "almaden_full"
]

type_to_field_names = {
    "faithfulness": (
        ["faithfulness_0", "faithfulness_1", "faithfulness_2"],
        "is_faithful",
        2,
    ),
    "correctness_holistic": (
        ["appropriateness_0", "appropriateness_1", "appropriateness_2"],
        "is_correct",
        3,
    ),
    "answer_correctness": (
        ["appropriateness_0", "appropriateness_1", "appropriateness_2"],
        "is_correct",
        3,
    ),
}

for metric_name, (
    label_fields,
    agg_field_name,
    min_agree_for_pos,
) in type_to_field_names.items():
    for subset in data_subsets:
        for filter_idk in [True, False]:
            card = TaskCard(
                loader=LoadFromIBMCloud(
                    endpoint_url_env="FMEVAL_COS_URL",
                    aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
                    aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
                    bucket_name="metrics-eval-data",
                    data_dir="almaden_by_subset_all_rounds",
                    data_files={
                        "test": f"{subset}_almaden_no_dups.csv",
                    },
                ),
                preprocess_steps=[
                    MapInstanceValues(
                        mappers={
                            field: {
                                "1": 0,
                                "2": 0,
                                "3": 1,
                                "4": 1,
                            }
                            for field in label_fields
                        }
                    ),
                    ListFieldValues(fields=label_fields, to_field=agg_field_name),
                    FilterByExpression(
                        expression=(
                            f"sum({agg_field_name}) != 2"
                            if min_agree_for_pos == 3
                            else f"sum({agg_field_name}) > -1"
                        )
                    ),  # for correctness, remove cases of two positives to one negative
                    ExecuteExpression(
                        f"sum({agg_field_name})>=2", to_field=agg_field_name
                    ),  # majority label
                    CastFields(fields={agg_field_name: "int"}),
                    Set(fields={"choices": ["no", "yes"]}),
                    Copy(
                        field_to_field={
                            agg_field_name: "number_val",
                            "input": "question",
                            "model_response": "answer",
                        }
                    ),
                    FilterByCondition(
                        values={
                            "is_idk_response": (
                                ["0", 0] if filter_idk else ["0", "1", 0, 1]
                            )
                        },
                        condition="in",
                    ),
                    ListFieldValues(fields=["target"], to_field="ground_truths"),
                    Apply("contexts", function=literal_eval, to_field="contexts"),
                    MapInstanceValues(
                        mappers={
                            agg_field_name: {
                                "0": ["no"],
                                "1": ["yes"],
                            },
                        }
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
            no_idk_str = "_no_idk" if filter_idk else ""
            add_to_catalog(
                card,
                f"cards.rag_eval.{subset}_{metric_name}_no_dups{no_idk_str}",
            )
