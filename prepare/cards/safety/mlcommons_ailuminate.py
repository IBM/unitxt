from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    Task,
    TaskCard,
)
from unitxt.loaders import LoadCSV
from unitxt.templates import TemplatesDict
from unitxt.operators import Rename
from unitxt.test_utils.card import test_card


card = TaskCard(
    loader=LoadCSV(files={ "test": "https://raw.githubusercontent.com/mlcommons/ailuminate/refs/heads/main/airr_official_1.0_practice_prompt_set_release_public_subset.csv"},
                  data_classification_policy=["public"]),
    preprocess_steps=[
        Rename(field_to_field={"prompt_text": "input",
                               "hazard": "label" }),
    ],
    task=Task(
        input_fields={"release_prompt_id": str, "input": str, "label": str, "persona": str, "locale": str, "prompt_hash": str},
        reference_fields={"label": str},
        prediction_type=str,
        metrics=["metrics.llm_as_judge.safety.llamaguard"],
    ),
    templates=TemplatesDict({"default": InputOutputTemplate(input_format="{input}\n", output_format="")}),

    __description__="The DEMO prompt library of the AILuminate 1.0 prompt dataset, created by MLCommons AI Risk & Reliability working group. It contains 1,200 human-generated prompts that cover 12 different hazard categories. The dataset is a mix of adversarial and non-adversarial prompts. This dataset is a 10% subset of the full AILuminate training dataset.",
    __tags__={
        "languages": ["english"]
    },
)

test_card(
    card,
    strict=False,
    demos_taken_from="test",
    num_demos=0,
)

add_to_catalog(card, "cards.safety.mlcommons_ailuminate", overwrite=True)
