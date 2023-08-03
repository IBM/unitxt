from datasets import load_dataset_builder
from prepare.cards.mmlu import (
    MMLU_TEMPLATES,
    multiple_choice_inputs_outputs,
    multiple_choice_preprocess,
)
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    CopyFields,
    IndexOf,
    JoinStr,
    RenameFields,
    TakeByField,
    ZipFieldValues,
)
from src.unitxt.test_utils.card import test_card

numbering = tuple(str(x) for x in range(200))
# numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
expected_answer = "number"  # 'number_and_answer' #'number'

card = TaskCard(
    loader=LoadHF(path="hellaswag"),
    preprocess_steps=[
        AddFields({"numbering": numbering}),
        IndexOf(search_in="numbering", index_of="label", to_field="index"),
        *multiple_choice_preprocess(
            question="ctx", numbering="numbering", choices="endings", topic="activity_label", label_index="index"
        ),
    ],
    task=FormTask(
        **multiple_choice_inputs_outputs(),
        metrics=["metrics.accuracy"],
    ),
    templates=MMLU_TEMPLATES,
)
test_card(card)
add_to_catalog(card, f"cards.hellaswag", overwrite=True)
