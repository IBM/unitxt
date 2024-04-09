from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import AddFields, IndexOf, RenameFields
from unitxt.test_utils.card import test_card

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
            AddFields({"context_type": "article"}),
        ],
        task="tasks.qa.multiple_choice.with_context",
        templates="templates.qa.multiple_choice.with_context.all",
    )
    if subset == "middle":
        test_card(card, strict=False)
    add_to_catalog(card, f"cards.race_{subset}", overwrite=True)
