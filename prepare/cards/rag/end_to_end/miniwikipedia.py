import json

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import CastFields, Copy, ListFieldValues, Set
from unitxt.splitters import SplitRandomMix
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="rag-datasets/rag-mini-wikipedia",
        name="question-answer",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            {
                "train": "test[70%]",
                "test": "test[30%]",
            }
        ),
        Copy(
            field_to_field={
                "question": "question",
                "id": "question_id",
            },
        ),
        Set(
            fields={
                "reference_context_ids": [],
                "reference_contexts": [],
                "is_answerable_label": True,
                "metadata_field": "",
            }
        ),
        Wrap(field="answer", inside="list", to_field="reference_answers"),
    ],
    task="tasks.rag.end_to_end",
    # templates=["templates.empty"],
    templates={"default": "templates.rag.end_to_end.json_predictions"},
)

wrong_answer = {
    "contexts": ["hi"],
    "is_answerable": True,
    "answer": "Don't know",
    "context_ids": ["id0"],
}
test_card(
    card,
    strict=True,
    full_mismatch_prediction_values=[json.dumps(wrong_answer)],
    debug=False,
    demos_taken_from="test",
    demos_pool_size=5,
)

add_to_catalog(
    card,
    "cards.rag.benchmark.miniwiki.en",
)

# Documents

card = TaskCard(
    loader=LoadHF(
        path="rag-datasets/rag-mini-wikipedia",
        name="text-corpus",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            {
                "train": "passages[100%]",
            }
        ),
        Cast(field="id", to="str"}),
        Copy(
            field_to_field={
                "id": "document_id",
            },
        ),
        Wrap(field="passage", inside="list", to_field="passages"),
        Set(
            fields={
                "metadata_field": "",
                "title": "",
            }
        ),
    ],
    task="tasks.rag.corpora",
    templates={
        "empty": InputOutputTemplate(
            input_format="",
            output_format="",
        ),
    },
)

add_to_catalog(
    card,
    "cards.rag.documents.miniwiki.en",
)
