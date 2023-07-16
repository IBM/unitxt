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

from src.unitxt.catalog import add_to_catalog

card = TaskCard(
        loader=LoadHF(path='super_glue', name='wsc'),
        preprocess_steps=[
            SmallNoTestSplitter(),
            MapInstanceValues(mappers={'label': {"0": 'True', "1": 'False'}}),
            AddFields(
            fields={
                'choices': ['True', 'False'],
            }
            ),
            NormalizeListFields(
                fields=['choices']
            ),
        ],
        task=FormTask(
            inputs=['choices', 'span1_text', 'span2_text'],
            outputs=['label'],
            metrics=['accuracy'],
        ),
        templates=TemplatesList([
            InputOutputTemplate(
                input_format="""
                    Given this sentence: {span1_text}, classify if this sentence: {span2_text} is {choices}.
                """.strip(),
                output_format='{label}',
            ),
        ])
    )

add_to_catalog(card, 'wsc', 'cards', overwrite=True)
