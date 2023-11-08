from datasets import get_dataset_config_names
configs = get_dataset_config_names('GEM/xlsum')  #the languages
# now configs is the list of all languages showing in the dataset

from src.unitxt.blocks import (
    FormTask,
    InputOutputTemplate,
    LoadHF,
    TaskCard,
    TemplatesList
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

langs = configs

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path='GEM/xlsum', name=lang),
        preprocess_steps=[
            # SplitRandomMix({"train": "train[95%]", "validation": "train[5%]", "test": "validation"}),
            # CopyFields(field_to_field=[["answers/text", "answer"]], use_query=True),
        ],
        task=FormTask(
            inputs=["text"],
            outputs=["target"],
            metrics=["metrics.rouge"],
        ),
        templates=TemplatesList(
            [
                InputOutputTemplate(input_format="{text}", output_format="{target}"),
            ]
        )
    )
    if lang == langs[0]:
        test_card(card, debug=True)
    add_to_catalog(card, f"cards.xlsum.{lang}", overwrite=True)
