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
        "dataset_info_tags": [
            "task_categories:text-classification",
            "task_categories:token-classification",
            "task_categories:question-answering",
            "task_ids:natural-language-inference",
            "task_ids:word-sense-disambiguation",
            "task_ids:coreference-resolution",
            "task_ids:extractive-qa",
            "annotations_creators:expert-generated",
            "language_creators:other",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:extended|other",
            "language:en",
            "license:other",
            "superglue",
            "NLU",
            "natural language understanding",
            "croissant",
            "arxiv:1905.00537",
            "region:us",
        ]
    },
)

test_card(card)
add_to_catalog(card, "cards.wsc", overwrite=True)
