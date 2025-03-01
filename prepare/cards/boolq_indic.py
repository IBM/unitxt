from unitxt.blocks import (
    LoadHF,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    CastFields,
    FilterByCondition,
    Rename,
)
from unitxt.test_utils.card import test_card

languages = ["bn", "gu", "hi", "kn", "mr", "ml", "or", "pa", "ta", "te"]

for language in languages:
    card = TaskCard(
        loader=LoadHF(path="sarvamai/boolq-indic"),
        preprocess_steps=[
            FilterByCondition(values={"language": language}, condition="eq"),
            "splitters.small_no_test",
            Set(
                {
                    "text_a_type": "passage",
                    "text_b_type": "question",
                    "classes": ["yes", "no"],
                    "type_of_relation": "answer",
                },
            ),
            CastFields(fields={"answer": "str"}),
            Rename(
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
            "annotations_creators": "Sarvam AI",
            "arxiv": "1905.10044",
            "license": "cc-by-sa-3.0",
            "multilinguality": "multilingual",
            "region": "in",
            "size_categories": "10K<n<100K",
            "source_datasets": "translation",
            "task_categories": "text-classification",
            "task_ids": "natural-language-inference",
        },
        __description__=(
            "A multilingual version of the BoolQ (Boolean Questions) dataset, translated from English into 10 Indian languages. It is a question-answering dataset for yes/no questions containing ~12k naturally occurring questions."
        ),
    )

    test_card(card, demos_taken_from="test")
    add_to_catalog(card, f"cards.boolq_indic.{language}.classification", overwrite=True)

    card = TaskCard(
        loader=LoadHF(path="sarvamai/boolq-indic"),
        preprocess_steps=[
            FilterByCondition(values={"language": language}, condition="eq"),
            "splitters.small_no_test",
            Set(
                {
                    "context_type": "passage",
                    "choices": ["yes", "no"],
                },
            ),
            CastFields(fields={"answer": "str"}),
            Rename(
                field_to_field={
                    "passage": "context",
                }
            ),
        ],
        task="tasks.qa.multiple_choice.with_context",
        templates="templates.qa.multiple_choice.with_context.all",
        __tags__={
            "annotations_creators": "Sarvam AI",
            "arxiv": "1905.10044",
            "license": "cc-by-sa-3.0",
            "multilinguality": "multilingual",
            "region": "in",
            "size_categories": "10K<n<100K",
            "source_datasets": "translation",
            "task_categories": "text-classification",
            "task_ids": "natural-language-inference",
        },
        __description__=(
            "A multilingual version of the BoolQ (Boolean Questions) dataset, translated from English into 10 Indian languages. It is a question-answering dataset for yes/no questions containing ~12k naturally occurring questions."
        ),
    )

    test_card(card, demos_taken_from="test", strict=False)
    add_to_catalog(
        card, f"cards.boolq_indic.{language}.multiple_choice", overwrite=True
    )
