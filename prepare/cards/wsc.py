from unitxt.blocks import InputOutputTemplate, LoadHF, TemplatesList
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.prepare_utils.card_types import add_classification_choices
from unitxt.task import FormTask
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="super_glue", name="wsc"),
    preprocess_steps=[
        "splitters.small_no_test",
        *add_classification_choices("label", {"0": "False", "1": "True"}),
    ],
    task=FormTask(
        inputs=["choices", "text", "span1_text", "span2_text"],
        outputs=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this sentence: {text} classify if "{span2_text}" refers to "{span1_text}".
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
    __tags__={
        "NLU": True,
        "annotations_creators": "expert-generated",
        "arxiv": "1905.00537",
        "croissant": True,
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "natural language understanding": True,
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "extended|other",
        "superglue": True,
        "task_categories": [
            "text-classification",
            "token-classification",
            "question-answering",
        ],
        "task_ids": [
            "natural-language-inference",
            "word-sense-disambiguation",
            "coreference-resolution",
            "extractive-qa",
        ],
    },
)

test_card(card)
add_to_catalog(card, "cards.wsc", overwrite=True)
