from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    CopyPasteFields
)
from src.unitxt.test_utils.card import test_card

from src.unitxt.catalog import add_to_catalog

card = TaskCard(
        loader=LoadHF(path='squad'),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'validation'}),
            CopyPasteFields({'answers/text': 'answer'}),
        ],
        task=FormTask(
            inputs=['context', 'question'],
            outputs=['answer'],
            metrics=['metrics.squad'],
        ),
        templates=TemplatesList([
            InputOutputTemplate(
                input_format="""
                    Given: {context}, answer {question}
                """.strip(),
                output_format='{answer}',
            ),
               ])
    )

test_card(card)
add_to_catalog(card, 'cards.squad', overwrite=True)


        