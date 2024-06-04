from unitxt.blocks import (
    AddFields,
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    CastFields,
    MapInstanceValues,
    RenameFields,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="google/boolq"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(
            {
                "text_a_type": "passage",
                "text_b_type": "question",
                "classes": ["yes", "no"],
                "type_of_relation": "answer",
            },
        ),
        CastFields(fields={"answer": "str"}),
        MapInstanceValues(mappers={"answer": {"True": "yes", "False": "no"}}),
        RenameFields(
            field_to_field={
                "passage": "text_a",
                "question": "text_b",
                "answer": "label",
            }
        ),
    ],
    task="tasks.classification.multi_class.relation",
    templates="templates.classification.multi_class.relation.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": "1905.10044",
        "flags": ["croissant"],
        "language": "en",
        "language_creators": "found",
        "license": "cc-by-sa-3.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": "natural-language-inference",
    },
    __description__=(
        "Dataset Card for Boolq\n"
        "Dataset Summary\n"
        "BoolQ is a question answering dataset for yes/no questions containing 15942 examples. These questions are naturally\n"
        "occurring ---they are generated in unprompted and unconstrained settings.\n"
        "Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context.\n"
        "The text-pair classification setup is similar to existing natural language inference tasks.\n"
        "Supported Tasks… See the full description on the dataset page: https://huggingface.co/datasets/google/boolq."
    ),
)

test_card(card, demos_taken_from="test")
add_to_catalog(card, "cards.boolq.classification", overwrite=True)

card = TaskCard(
    loader=LoadHF(path="google/boolq"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields(
            {
                "context_type": "passage",
                "choices": ["yes", "no"],
            },
        ),
        CastFields(fields={"answer": "str"}),
        MapInstanceValues(mappers={"answer": {"True": "yes", "False": "no"}}),
        RenameFields(
            field_to_field={
                "passage": "context",
            }
        ),
    ],
    task="tasks.qa.multiple_choice.with_context",
    templates="templates.qa.multiple_choice.with_context.all",
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": "1905.10044",
        "flags": ["croissant"],
        "language": "en",
        "language_creators": "found",
        "license": "cc-by-sa-3.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": "natural-language-inference",
    },
    __description__=(
        "Dataset Card for Boolq\n"
        "Dataset Summary\n"
        "BoolQ is a question answering dataset for yes/no questions containing 15942 examples. These questions are naturally\n"
        "occurring ---they are generated in unprompted and unconstrained settings.\n"
        "Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context.\n"
        "The text-pair classification setup is similar to existing natural language inference tasks.\n"
        "Supported Tasks… See the full description on the dataset page: https://huggingface.co/datasets/google/boolq."
    ),
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(card, "cards.boolq.multiple_choice", overwrite=True)
