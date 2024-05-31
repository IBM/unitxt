from unitxt.blocks import InputOutputTemplate, LoadHF, TemplatesList
from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.prepare_utils.card_types import add_classification_choices
from unitxt.task import Task
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="super_glue", name="wsc"),
    preprocess_steps=[
        "splitters.small_no_test",
        *add_classification_choices("label", {"0": "False", "1": "True"}),
    ],
    task=Task(
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
        "annotations_creators": "expert-generated",
        "arxiv": "1905.00537",
        "language": "en",
        "language_creators": "other",
        "license": "other",
        "multilinguality": "monolingual",
        "region": "us",
        "singletons": [
            "NLU",
            "croissant",
            "natural language understanding",
            "superglue",
        ],
        "size_categories": "10K<n<100K",
        "source_datasets": "extended|other",
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
    __description__=(
        "SuperGLUE (https://super.gluebenchmark.com/) is a new benchmark styled after\n"
        "GLUE with a new set of more difficult language understanding tasks, improved\n"
        "resources, and a new public leaderboard."
    ),
)

test_card(card)
add_to_catalog(card, "cards.wsc", overwrite=True)
