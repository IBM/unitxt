from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues
)
from src.unitxt.test_utils.card import test_card

from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import MultipleChoiceInputOutputTemplate

sub_task = 'abstract_algebra'
card = TaskCard(
    loader=LoadHF(path='cais/mmlu', name=sub_task),
    preprocess_steps=[
        RenameFields({'answer': 'label', 'question': 'label'})
    ],
    task=FormTask(
        inputs=['choices', 'question'],
        outputs=['label'],
        metrics=['metrics.accuracy'],
    ),
    templates=MultipleChoiceInputOutputTemplate([
        InputOutputTemplate(
            input_format="""
                    Question: {sentence1}.\nChoose one label from {labels}\nAnswers: {choices}. 
                """.strip(),
            output_format='{label}',
        ),
    ])
)

test_card(card)
add_to_catalog(card, f'cards.mmlu.{sub_task}', overwrite=True)
