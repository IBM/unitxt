
from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    SequentialRecipe,
    MapInstanceValues,
    FormTask,
    RenderAutoFormatTemplate,
)

from src.unitxt.catalog import add_to_catalog
from src.unitxt.load import load_dataset
from src.unitxt.text_utils import print_dict

recipe = SequentialRecipe(
    steps=[
        LoadHF(
            path='glue',
            name='wnli',
        ),
        SplitRandomMix(
            mix={
                'train': 'train[95%]',
                'validation': 'train[5%]',
                'test': 'validation',
            }
        ),
        MapInstanceValues(
            mappers={
                'label': {"0": 'entailment', "1": 'not_entailment'}
            }
        ),
        AddFields(
            fields={
                'choices': ['entailment', 'not_entailment'],
                'instruction': 'classify the relationship between the two sentences from the choices.',
            }
        ),
        FormTask(
            inputs=['choices', 'instruction', 'sentence1', 'sentence2'],
            outputs=['label'],
            metrics=['accuracy'],
        ),
        RenderAutoFormatTemplate(),
    ]
)

add_to_catalog(recipe, 'wnli', collection='recipes', overwrite=True)

dataset = load_dataset('wnli')

print_dict(dataset['train'][0])
