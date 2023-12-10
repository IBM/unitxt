from prepare.cards.mmlu import (
    multiple_choice_inputs_outputs,
    multiple_choice_preprocess,
)
from src.unitxt.blocks import InputOutputTemplate, LoadHF, TemplatesList
from src.unitxt.card import TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import AddFields, ListFieldValues
from src.unitxt.task import FormTask
from src.unitxt.test_utils.card import test_card

numbering = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
card = TaskCard(
    loader=LoadHF(path="super_glue", name="copa"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields({"numbering": numbering, "topic": "commonsense causal reasoning"}),
        ListFieldValues(fields=["choice1", "choice2"], to_field="choices"),
        *multiple_choice_preprocess(
            context="premise",
            question="question",
            numbering="numbering",
            choices="choices",
            topic="topic",
            label_index="label",
        ),
    ],
    task=FormTask(
        **multiple_choice_inputs_outputs(context=True),
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    What was the {sentence1} of the following:\n{context}\nAnswers: {choices}\nAnswer:
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.copa", overwrite=True)
