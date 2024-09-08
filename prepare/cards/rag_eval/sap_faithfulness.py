from ast import literal_eval

from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import Apply, CastFields, Copy, FilterByCondition, Set
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card


def add_sap_faithfulness_card(filter_idk=True, answer_field="model_prediction"):
    first_preprocess_steps = (
        [FilterByCondition(values={"is_idk_response": 0}, condition="eq")]
        if filter_idk
        else []
    )
    card = TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="SAP",
            data_files={
                "test": "SAP_Human_Evaluations_jan_march_2024_per_example_with_idk_informational_response.csv",
            },
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            *first_preprocess_steps,
            FilterByCondition(values={"is_faithful": [0, 1]}, condition="in"),
            Set(fields={"choices": ["no", "yes"]}),
            CastFields(fields={"is_faithful": "int"}),
            Copy(
                field_to_field={
                    "is_faithful": "number_val",
                    "input": "question",
                    "input_background": "contexts",
                    answer_field: "answer",
                }
            ),
            Apply("contexts", function=literal_eval, to_field="contexts"),
            MapInstanceValues(
                mappers={"is_faithful": {str(0): ["no"], str(1): ["yes"]}},
                strict=False,
            ),
            Copy(field_to_field={"is_faithful": "textual_label"}),
        ],
        task="tasks.rag_eval.faithfulness.binary",
        templates="templates.rag_eval.faithfulness.all",
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
        card,
        f"cards.rag_eval.sap_faithfulness{'_no_idk' if filter_idk else ''}",
    )


for filter_idk in [True, False]:
    for answer_field in ["model_prediction"]:
        add_sap_faithfulness_card(filter_idk=filter_idk, answer_field=answer_field)
