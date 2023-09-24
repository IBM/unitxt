from src.unitxt.blocks import LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import AddFields, IndexOf, RenameFields
from src.unitxt.test_utils.card import test_card

numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

for subset in ["all", "high", "middle"]:
    card = TaskCard(
        loader=LoadHF(path="race", name=subset),
        preprocess_steps=[
            AddFields({"numbering": numbering}),
            IndexOf(search_in="numbering", index_of="answer", to_field="answer"),
            RenameFields(
                field_to_field={"options": "choices", "article": "context"},
            ),
        ],
        task="tasks.qa.multiple_choice.contextual",
        templates="templates.qa.multiple_choice.contextual.all",
    )
    test_card(card)
    add_to_catalog(card, f"cards.race_{subset}", overwrite=True)
