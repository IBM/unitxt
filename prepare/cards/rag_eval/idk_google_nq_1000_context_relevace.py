from ast import literal_eval

from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import Apply, CastFields, Copy, Set
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card


def get_card(contexts_field="contexts"):
    return TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="idk_google_nq",
            data_files={
                "test": "google_nq_syn_1000_pairs.csv",
            },
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            Set(fields={"choices": ["no", "yes"]}),
            CastFields(fields={"is_context_relevant": "int"}),
            Copy(
                field_to_field={
                    "is_context_relevant": "number_val",
                    contexts_field: "contexts",
                }
            ),
            Apply("contexts", function=literal_eval, to_field="contexts"),
            MapInstanceValues(
                mappers={"is_context_relevant": {str(0): ["no"], str(1): ["yes"]}},
                strict=False,
            ),
            Copy(field_to_field={"is_context_relevant": "textual_label"}),
        ],
        task="tasks.rag_eval.context_relevance.binary",
        templates="templates.rag_eval.context_relevance.all",
        sampler=DiverseLabelsSampler(
            choices="choices", labels="textual_label", include_empty_label=False
        ),
    )


card = get_card()
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
    "cards.rag_eval.idk_google_nq_1000_context_relevance",
)
