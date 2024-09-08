import sys

import numpy as np
import unitxt
from fm_eval.runnables.local_catalogs import add_to_private_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import (
    CastFields,
    Copy,
    ListFieldValues,
    Set,
    Shuffle,
)
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card

unitxt.settings.seed = 42


def get_card(source):
    file_name = "all.csv" if source == "all" else f"{source}_download.csv"
    return TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="TRUE/data",
            data_files={"test": file_name},
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            Shuffle(page_size=sys.maxsize),
            Copy(field_to_field={"label": "is_faithful", "generated_text": "answer"}),
            MapInstanceValues(mappers={"answer": {str(np.nan): ""}}, strict=False),
            ListFieldValues(fields=["grounding"], to_field="contexts"),
            Set(
                fields={
                    "question": "-",
                    "choices": ["no", "yes"],
                }
            ),
            CastFields(fields={"is_faithful": "int"}),
            Copy(field_to_field={"is_faithful": "number_val"}),
            MapInstanceValues(
                mappers={"is_faithful": {str(0): ["no"], str(1): ["yes"]}}
            ),
        ],
        task="tasks.rag_eval.faithfulness.binary",
        templates="templates.rag_eval.faithfulness.all",
        sampler=DiverseLabelsSampler(
            choices="choices", labels="is_faithful", include_empty_label=False
        ),
    )


TRUE_SUBSETS = [
    "qags_cnndm",
    "qags_xsum",
    "frank_valid",
    "begin_dev",
    "q2",
    "summeval",
    "mnbm",
    "paws",
    "dialfact_valid",
    "fever_dev",
    "vitc_dev",
    "true_all_subsets",
]


for source in TRUE_SUBSETS:
    card = get_card(source)
    test_card(
        card,
        num_demos=2,
        strict=False,
        demos_removed_from_data=False,
        demos_taken_from="test",
        demos_pool_size=20,
    )
    add_to_private_catalog(
        card,
        f"cards.rag_eval.true_{source}_faithfulness",
    )
