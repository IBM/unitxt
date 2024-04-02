import sys

from src.unitxt import add_to_catalog
from src.unitxt.card import TaskCard
from src.unitxt.collections_operators import DuplicateByList
from src.unitxt.loaders import LoadHF
from src.unitxt.operators import (
    AddFields,
    ExecuteExpression,
    FilterByCondition,
    IndexOf,
    ListFieldValues,
    RenameFields,
    Shuffle,
)
from src.unitxt.test_utils.card import test_card

gec_card = TaskCard(
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

error_detection_metrics = [
    "metrics.accuracy",
    "metrics.f1_binary",
    "metrics.precision_binary",
    "metrics.recall_binary",
]

error_detection_card = TaskCard(
    loader=LoadHF(path="grammarly/coedit", streaming=True),
    preprocess_steps=[
        FilterByCondition(values={"task": "gec"}, condition="eq"),
        "splitters.small_no_test",
        ExecuteExpression(expression="': '.join(src.split(': ')[1:])", to_field="src"),
        ListFieldValues(
            fields=["tgt", "src"],
            to_field="correct_and_incorrect",
        ),
        DuplicateByList(field="correct_and_incorrect", to_field="text"),
        IndexOf(index_of="text", search_in="correct_and_incorrect", to_field="label"),
        AddFields(
            fields={
                "class": "Grammatically incorrect",
                "text_type": "text",
            }
        ),
        Shuffle(page_size=sys.maxsize),
    ],
    task=f"tasks.classification.binary[metrics=[{','.join(error_detection_metrics)}]]",
    templates=["templates.grammatical_error_detection.yes_no"],
)


test_card(gec_card, strict=False)
add_to_catalog(gec_card, "cards.coedit_gec", overwrite=True)

test_card(error_detection_card)
add_to_catalog(error_detection_card, "cards.coedit_error_detection", overwrite=True)
