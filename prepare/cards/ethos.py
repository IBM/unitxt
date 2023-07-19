import os
from pathlib import Path

from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues
)
from unitxt.card import ClassificationCard

from unitxt.catalog import add_to_catalog

from unitxt.instructions import TextualInstruction
from unitxt.templates import outputs_inputs2templates, instructions2templates

# TODO convert str to instructions
from unitxt.test_utils.card import test_card

classification_instructions = ["Predict the class of the following ({choices}):",
                               "What is the type of the following? Types:{choices}",
                               "Which of the choices, best describes the following text:"]  # TODO How to share
# instructions?
classification_instructions = [TextualInstruction(text=x) for x in classification_instructions]

instructions = classification_instructions
card = ClassificationCard(
    loader=LoadHF(path='ethos'),
    label_name="label",
    label2string={"0": 'hate speech', "1": 'not hate speech'},
    inputs=['text'],
    metrics=['metrics.accuracy'],
    preprocess_steps=[
        "splitter.single_split_test"
    ],
    templates=instructions2templates(
        instructions=instructions,
        templates=[
            InputOutputTemplate(
                input_format="""
                    {instruction} Sentence: {text}. Choices {choices}
                """.strip(),
                output_format='{label}',
            ),
        ]
    ) +
              outputs_inputs2templates(inputs=[
                  """Given this sentence: {text}. Classify if it contains hatespeech. Choices: {choices}.""",
                  """Does the following sentence contains hatespeech? Answer by choosing one of the options {choices}. sentence: {text}."""],
                  outputs='{label}'
              )
)

project_dir = Path(__file__).parent.parent.parent.absolute()
catalog_dir = os.path.join(project_dir, 'fm_eval', 'catalogs', 'private')
test_card(card)
add_to_catalog(card, 'cards.ethos', overwrite=True, catalog_path=catalog_dir)
