from src.unitxt.blocks import InputOutputTemplate, LoadHF, TemplatesList
from src.unitxt.catalog import add_to_catalog
from src.unitxt.prepare_utils.card_types import create_sentence_classification_card
from src.unitxt.splitters import RenameSplits
from src.unitxt.templates import outputs_inputs2templates
from src.unitxt.test_utils.card import test_card

card = create_sentence_classification_card(
    loader=LoadHF(path="ethos", name="binary"),
    label_name="label",
    label2string={"0": "not hate speech", "1": "hate speech"},
    inputs=["text"],
    metrics=["metrics.accuracy"],
    preprocess_steps=[RenameSplits(mapper={"train": "test"}), "splitters.test_to_train_dev"],
    templates=outputs_inputs2templates(
        inputs=[
            """Given this sentence: {sentence1}. Classify if it contains hatespeech. Choices: {choices}.""",
            """Does the following sentence contains hatespeech? Answer by choosing one of the options {choices}. sentence: {sentence1}.""",
        ],
        outputs="{label}",
    ),
)

test_card(card, tested_split="test")
add_to_catalog(card, "cards.ethos_binary", overwrite=True)
