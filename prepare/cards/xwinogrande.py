from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import AddConstant, CastFields, ListFieldValues, RenameFields
from src.unitxt.test_utils.card import test_card

for lang in ["pt", "ru", "zh", "en", "jp"]:
    card = TaskCard(
        loader=LoadHF(path="Muennighoff/xwinograd", name=lang),
        preprocess_steps=[
            ListFieldValues(fields=["option1", "option2"], to_field="choices"),
            CastFields(fields={"answer": "int"}),
            AddConstant(field="answer", add=-1),
            RenameFields(
                field_to_field={"sentence": "question"},
            ),
        ],
        task="tasks.qa.multiple_choice.original",
        templates="templates.qa.multiple_choice.all",
    )
    if lang == "pt":
        test_card(card, demos_taken_from="test")
    add_to_catalog(card, f"cards.xwinogrande.{lang}", overwrite=True)
