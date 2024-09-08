import sys
from ast import literal_eval

import unitxt
from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import Apply, CastFields, Copy, Set, Shuffle
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card

unitxt.settings.seed = 42


def get_card(contexts_field="contexts"):
    return TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="squad2",
            data_files={
                "test": "squad2_context_relevance_with_sents_and_full.csv",
            },
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            Shuffle(page_size=sys.maxsize),
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


context_field_to_suf = {
    "contexts": "",
    "all_contexts": "_with_sents",
    "contexts_sents": "_sents_only",
}
for context_field, suf in context_field_to_suf.items():
    card = get_card(context_field)
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
        f"cards.rag_eval.squad2_context_relevance{suf}",
    )
