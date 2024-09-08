from ast import literal_eval

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import Apply, Copy, Set
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card

type_to_field_name = {
    "faithfulness": "is_faithful",
    "correctness_holistic": "is_correct",
    "context_relevance": "is_context_relevant",
}

for metric_name, field_name in type_to_field_name.items():
    card = TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="AutoRAG",
            data_files={"test": "autorag_data.csv"},
        ),
        preprocess_steps=[
            Apply("contexts", function=literal_eval, to_field="contexts"),
            Set(fields={field_name: ["yes"], "number_val": 1}),  # mock labels
            Set(fields={"choices": ["no", "yes"]}),
            Copy(field_to_field={field_name: "textual_label"}),
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
    add_to_catalog(card, f"cards.rag_eval.autorag_data_{metric_name}", overwrite=True)
