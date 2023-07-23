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
from unitxt.operators import MapNestedDictValuesByQueries

card = TaskCard(
        loader=LoadHF(path='squad'),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[95%]', 'validation': 'train[5%]', 'test': 'validation'}),
            MapNestedDictValuesByQueries(field_to_query={'answer': 'answers/text'}),
            #MapInstanceValues(mappers={'label': {"0": 'entailment', "1": 'not_entailment'}}),
            #AddFields(
            #fields={
            #    'choices': ['entailment', 'not_entailment'],
            #}
          #  ),
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


        