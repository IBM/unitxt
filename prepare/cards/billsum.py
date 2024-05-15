from unitxt import add_to_catalog
from unitxt.blocks import AddFields, TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import RenameFields
from unitxt.test_utils.card import test_card

# https://huggingface.co/datasets/billsum
card = TaskCard(
    loader=LoadHF(path="billsum"),
    preprocess_steps=[
        RenameFields(field_to_field={"text": "document"}),
        AddFields(fields={"document_type": "document"}),
    ],
    task="tasks.summarization.abstractive",
    templates="templates.summarization.abstractive.all",
)
test_card(
    card,
    format="formats.textual_assistant",
)
add_to_catalog(card, "cards.billsum", overwrite=True)
