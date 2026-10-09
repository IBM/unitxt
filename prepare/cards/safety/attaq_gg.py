from unitxt import add_to_catalog
from unitxt.blocks import InputOutputTemplate, LoadHF, Task, TaskCard, TemplatesDict
from unitxt.operators import Shuffle
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="ibm/AttaQ"),
    preprocess_steps=[
        RenameSplits(mapper={"train": "test"}),
        Shuffle(page_size=2800),
    ],
    task=Task(
        input_fields={"input": str},
        reference_fields={"label": str},
        prediction_type=str,
        metrics=[
            "metrics.granite_guardian.assistant_risk.harm[prediction_type=str,user_message_field=input,assistant_message_field=prediction]",
        ],
    ),
    templates=TemplatesDict(
        {"default": InputOutputTemplate(input_format="{input}", output_format="")}
    ),
)

test_card(card, strict=False, demos_taken_from="test", num_demos=0)
add_to_catalog(card, "cards.safety.attaq_gg", overwrite=True)
