import json

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.loaders import LoadHF
from unitxt.operators import Cast, Copy, Set
from unitxt.splitters import RenameSplits
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="enelpol/rag-mini-bioasq",
        name="question-answer-passages",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                "question": "question",
                "id": "question_id",
            },
        ),
        Cast(
            field="relevant_passage_ids",
            to="str",
            to_field="reference_context_ids",
            process_every_value=True,
        ),
        Set(
            fields={
                "reference_contexts": [],
                "is_answerable_label": True,
                "metadata_field": "",
            }
        ),
        Wrap(
            field="answer",
            inside="list",
            to_field="reference_answers",
        ),
    ],
    task="tasks.rag.end_to_end",
    templates={"default": "templates.rag.end_to_end.json_predictions"},
    __tags__={"license": "cc-by-2.5"},
    __description__="""This dataset is a subset of a training dataset by the BioASQ Challenge, which is available here.

It is derived from rag-datasets/rag-mini-bioasq.

Modifications include:

filling in missing passages (some of them contained "nan" instead of actual text),
changing relevant_passage_ids' type from string to sequence of ints,
deduplicating the passages (removed 40 duplicates) and fixing the relevant_passage_ids in QAP triplets to point to the corrected, deduplicated passages' ids,
splitting QAP triplets into train and test splits.
""",
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

add_to_catalog(card, "cards.rag.benchmark.bioasq.en", overwrite=True)


# Documents
card = TaskCard(
    loader=LoadHF(
        path="enelpol/rag-mini-bioasq",
        name="text-corpus",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        RenameSplits({"test": "train"}),
        Cast(field="id", to="str"),
        Copy(field="id", to_field="document_id"),
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
    __tags__={"license": "cc-by-2.5"},
    __description__="""This dataset is a subset of a training dataset by the BioASQ Challenge, which is available here.

It is derived from rag-datasets/rag-mini-bioasq.

Modifications include:

filling in missing passages (some of them contained "nan" instead of actual text),
changing relevant_passage_ids' type from string to sequence of ints,
deduplicating the passages (removed 40 duplicates) and fixing the relevant_passage_ids in QAP triplets to point to the corrected, deduplicated passages' ids,
splitting QAP triplets into train and test splits.
""",
)

# Not testing card, because documents are not evaluated.
add_to_catalog(card, "cards.rag.documents.bioasq.en", overwrite=True)
