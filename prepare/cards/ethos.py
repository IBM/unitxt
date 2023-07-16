from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    ClassificationCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues
)

from src.unitxt.catalog import add_to_catalog

from unitxt.templates import outputs_inputs2templates, instructions2templates

# TODO convert str to instructions
Classification_instructions = ["Predict the class of the following ({choices}):",
                               "What is the type of the following? Types:{choices}",
                               "Which of the choices, best describes the following text:"]  # TODO How to share instructions?

instructions = [Classification_instructions]
card = ClassificationCard(
    loader=LoadHF(path='ethos'),
    label="label",
    label2string={"0": 'hate speech', "1": 'not hate speech'},
    inputs=['text'],
    outputs=['label'],
    metrics=['accuracy'],
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
add_to_catalog(card, 'ethos-test-only', 'cards', overwrite=True)
load_dataset()