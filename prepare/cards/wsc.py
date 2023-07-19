import os
from pathlib import Path

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
from unitxt.card import ClassificationCard

from unitxt.test_utils.card import test_card

card = ClassificationCard(
    loader=LoadHF(path='glue', name='wsc'),
    preprocess_steps=[
        'splitters.small_no_test', ],
    label_name="label",
    label2string={"0": 'True', "1": 'False'},
    inputs=['span1_text', 'span2_text'],
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
add_to_catalog(card, 'cards.wsc', overwrite=True,catalog_path=catalog_dir)
