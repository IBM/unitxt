from src.unitxt.blocks import (
    CopyFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="squad"),
    preprocess_steps=[
        SplitRandomMix({"train": "train[95%]", "validation": "train[5%]", "test": "validation"}),
        CopyFields(field_to_field=[["answers/text", "answer"]], use_query=True),
    ],
    task=FormTask(
        inputs=["context", "question"],
        outputs=["answer"],
        metrics=["metrics.squad"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given: {context}, answer {question}
                """.strip(),
                output_format="{answer}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.squad", overwrite=True)
