from prepare.cards.mmlu import (
    multiple_choice_inputs_outputs,
    multiple_choice_preprocess,
)
from src.unitxt.blocks import AddFields, FormTask, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import IndexOf, RenameFields
from src.unitxt.test_utils.card import test_card

# numbering=tuple(str(x) for x in range(200))
numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
expected_answer = "number"  # 'number_and_answer' #'number'

card = TaskCard(
    loader=LoadHF(path="openbookqa"),
    preprocess_steps=[
        AddFields(
            {
                "topic": "general continuation",
                "numbering": numbering,
            },
        ),
        RenameFields(
            field_to_field={"choices/text": "text", "choices/label": "numbering"},
            use_query=True,
        ),
        IndexOf(search_in="numbering", index_of="answerKey", to_field="index"),
        *multiple_choice_preprocess(
            question="question_stem",
            numbering="numbering",
            choices="text",
            topic="topic",
            label_index="index",
        ),
    ],
    task=FormTask(
        **multiple_choice_inputs_outputs(),
        metrics=["metrics.accuracy"],
    ),
    templates="templates.qa.multiple_choice.original.all",
)
test_card(card)
add_to_catalog(card, "cards.openbookQA", overwrite=True)
