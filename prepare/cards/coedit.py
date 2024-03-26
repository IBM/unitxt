from src.unitxt import add_to_catalog
from src.unitxt.card import TaskCard
from src.unitxt.loaders import LoadHF
from src.unitxt.operators import (
    ExecuteExpression,
    FilterByCondition,
    ListFieldValues,
    RenameFields,
)
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="grammarly/coedit", streaming=True),
    preprocess_steps=[
        FilterByCondition(values={"task": "gec"}, condition="eq"),
        "splitters.small_no_test",
        ExecuteExpression(expression="': '.join(src.split(': ')[1:])", to_field="src"),
        RenameFields(field_to_field={"src": "original_text"}),
        ListFieldValues(fields=["tgt"], to_field="corrected_texts"),
    ],
    task="tasks.grammatical_error_correction",
    templates=["templates.grammatical_error_correction.simple"],
)


test_card(card, strict=False)
add_to_catalog(card, "cards.coedit_gec", overwrite=True)
