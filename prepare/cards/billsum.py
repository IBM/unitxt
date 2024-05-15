from unitxt import add_to_catalog
from unitxt.blocks import AddFields, FormTask, TaskCard
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
    task=FormTask(
        inputs=["document", "document_type"],
        outputs=["summary"],
        metrics=["metrics.rouge"],
        augmentable_inputs=["document"],
    ),
    templates="templates.summarization.abstractive.all",
)
test_card(
    card,
    format="formats.textual_assistant",
)
add_to_catalog(card, "cards.billsum", overwrite=True)
