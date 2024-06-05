from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import AddConstant, CastFields, ListFieldValues, RenameFields
from unitxt.test_utils.card import test_card

for subtask in ["debiased", "l", "m", "s", "xl", "xs"]:
    card = TaskCard(
        loader=LoadHF(path="winogrande", name=f"winogrande_{subtask}"),
        preprocess_steps=[
            "splitters.small_no_test",
            ListFieldValues(fields=["option1", "option2"], to_field="choices"),
            CastFields(fields={"answer": "int"}),
            AddConstant(field="answer", add=-1),
            RenameFields(
                field_to_field={"sentence": "question"},
            ),
        ],
        task="tasks.qa.multiple_choice.open",
        templates="templates.qa.multiple_choice.open.all",
        __tags__={"language": "en", "region": "us"},
        __description__=(
            "WinoGrande is a new collection of 44k problems, inspired by Winograd Schema Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and robustness against the dataset-specific bias. Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right option for a given sentence which requires commonsense reasoning… See the full description on the dataset page: https://huggingface.co/datasets/winogrande"
        ),
    )
    if subtask == "debiased":
        test_card(card, demos_taken_from="test", strict=False)
    add_to_catalog(
        card, f"cards.winogrande.{subtask.replace('-', '_')}", overwrite=True
    )
