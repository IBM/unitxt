import os
from pathlib import Path
from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues
)
from src.unitxt.test_utils.card import test_card

from src.unitxt.catalog import add_to_catalog
from unitxt.card import ClassificationCard

card = ClassificationCard(
    loader=LoadHF(path='glue', name='wnli'),
    preprocess_steps=[
        'splitters.small_no_test', ],
    label_name="label",
    label2string={"0": 'entailment', "1": 'not entailment'},
    inputs=['text'],
    metrics=['metrics.accuracy'],
    templates=TemplatesList([
        InputOutputTemplate(
            input_format="""
                    Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
                """.strip(),
            output_format='{label}',
        ),
    ])
)



project_dir = Path(__file__).parent.parent.parent.absolute()
catalog_dir = os.path.join(project_dir, 'fm_eval', 'catalogs', 'private')
test_card(card)
add_to_catalog(card, 'cards.wnli', overwrite=True,catalog_path=catalog_dir)
