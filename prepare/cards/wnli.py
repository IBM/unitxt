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

card = ClassificationCard(
    loader=LoadHF(path='glue', name='wnli'),
    preprocess_steps=[
        'splitters.small_no_test', ],
    label="label",
    label2string={"0": 'entailment', "1": 'not entailment'},
    inputs=['text'],
    outputs=['label'],
    metrics=['accuracy'],
    templates=TemplatesList([
        InputOutputTemplate(
            input_format="""
                    Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
                """.strip(),
            output_format='{label}',
        ),
    ])
)

add_to_catalog(card, 'wnli_card', 'cards', overwrite=True)
