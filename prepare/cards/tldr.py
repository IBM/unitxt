from unitxt import add_to_catalog
from unitxt.blocks import AddFields, SplitRandomMix, TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import RenameFields
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="webis/tldr-17", streaming=True),
    preprocess_steps=[
        SplitRandomMix({"train": "train[50%]", "test": "train[50%]"}),
        RenameFields(field_to_field={"content": "document"}),
        AddFields(fields={"document_type": "document"}),
    ],
    task="tasks.summarization.abstractive",
    templates="templates.summarization.abstractive.all",
)
test_card(
    card,
    format="formats.textual_assistant",
)
add_to_catalog(card, "cards.tldr", overwrite=True)
