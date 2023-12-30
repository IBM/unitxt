from datasets import get_dataset_config_names

from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    RenameFields,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

configs = get_dataset_config_names("GEM/xlsum")  # the languages
# now configs is the list of all languages showing in the dataset


langs = configs

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="GEM/xlsum", name=lang),
        preprocess_steps=[
            RenameFields(field_to_field={"text": "document", "target": "summary"}),
            AddFields(fields={"document_type": "document"}),
        ],
        task=FormTask(
            inputs=["document", "document_type"],
            outputs=["summary"],
            metrics=["metrics.rouge"],
        ),
        templates=TemplatesList(
            [
                InputOutputTemplate(
                    input_format="{document}", output_format="{summary}"
                ),
            ]
        ),
    )
    if lang == langs[0]:
        test_card(card, debug=False, strict=False)
    add_to_catalog(card, f"cards.xlsum.{lang}", overwrite=True)
