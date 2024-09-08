import sys

import numpy as np
import unitxt
from unitxt import add_to_catalog
from unitxt.blocks import MapInstanceValues, TaskCard
from unitxt.loaders import LoadFromIBMCloud
from unitxt.operators import (
    CastFields,
    Copy,
    FilterByCondition,
    ListFieldValues,
    Set,
    Shuffle,
)
from unitxt.splitters import DiverseLabelsSampler
from unitxt.test_utils.card import test_card

unitxt.settings.seed = 42


def get_card(label_col, answer_field="Response"):
    return TaskCard(
        loader=LoadFromIBMCloud(
            endpoint_url_env="FMEVAL_COS_URL",
            aws_access_key_id_env="FMEVAL_COS_ACCESS_KEY_ID",
            aws_secret_access_key_env="FMEVAL_COS_SECRET_ACCESS_KEY",
            bucket_name="metrics-eval-data",
            data_dir="ask_hr",
            data_files={
                "test": "UAT_combined_with_shortened_docs_annotated_informational_response.csv"
            },
        ),
        preprocess_steps=[
            FilterByCondition(values={"is_idk": [0, "0"]}, condition="in"),
            Shuffle(page_size=sys.maxsize),
            Copy(
                field_to_field={
                    label_col: "is_faithful",
                    answer_field: "answer",
                    "Query": "question",
                }
            ),
            FilterByCondition(values={answer_field: None}, condition="ne"),
            MapInstanceValues(mappers={"answer": {str(np.nan): ""}}, strict=False),
            ListFieldValues(fields=["contexts"], to_field="contexts"),
            Set(
                fields={
                    "choices": ["no", "yes"],
                }
            ),
            CastFields(fields={"is_faithful": "int"}),
            Copy(field_to_field={"is_faithful": "number_val"}),
            MapInstanceValues(
                mappers={"is_faithful": {str(0): ["no"], str(1): ["yes"]}}
            ),
            Copy(field_to_field={"is_faithful": "textual_label"}),
        ],
        task="tasks.rag_eval.faithfulness.binary",
        templates="templates.rag_eval.faithfulness.all",
        sampler=DiverseLabelsSampler(
            choices="choices", labels="textual_label", include_empty_label=False
        ),
    )


tested_dimensions = [
    "is_not_misunderstading_or_hallucination",
    "is_not_misunderstanding",
    "is_not_hallucination",
]
for label_col in tested_dimensions:
    for answer_field in ["Response", "informational_answer"]:
        informational_str = (
            "_informational" if answer_field == "informational_answer" else ""
        )
        file_suffix = (
            "" if label_col == "is_faithful" else label_col.replace("is_not", "")
        )
        card = get_card(label_col, answer_field=answer_field)
        test_card(
            card,
            num_demos=0,
            strict=False,
        )
        add_to_catalog(
            card,
            f"cards.rag_eval.askhr_single_turn_faithfulness{file_suffix}{informational_str}",
        )
