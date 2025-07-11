from unitxt import add_to_catalog
from unitxt.blocks import (
    InputOutputTemplate,
    Task,
    TaskCard,
)
from unitxt.loaders import LoadCSV
from unitxt.templates import TemplatesDict
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadCSV(
        files={
            "test": "https://raw.githubusercontent.com/mlcommons/ailuminate/0fe054f3d34209a0a1acf1b95174c5e8926f5c57/airr_official_1.0_practice_prompt_set_release_public_subset.csv"
        },  # pragma: allowlist secret
        data_classification_policy=["public"],
    ),
    task=Task(
        input_fields={
            "release_prompt_id": str,
            "prompt_text": str,
            "hazard": str,
            "persona": str,
            "locale": str,
            "prompt_hash": str,
        },
        reference_fields={},
        prediction_type=str,
        metrics=["metrics.llm_as_judge.safety.llamaguard"],
    ),
    templates=TemplatesDict(
        {
            "default": InputOutputTemplate(
                input_format="{prompt_text}\n", output_format=""
            )
        }
    ),
    __description__="The DEMO prompt library of the AILuminate 1.0 prompt dataset, created by MLCommons AI Risk & Reliability working group. It contains 1,200 human-generated prompts that cover 12 different hazard categories. The dataset is a mix of adversarial and non-adversarial prompts. This dataset is a 10% subset of the full AILuminate training dataset.",
    __tags__={"languages": ["english"]},
)

test_card(
    card,
    strict=False,
    demos_taken_from="test",
    num_demos=0,
)

add_to_catalog(card, "cards.safety.mlcommons_ailuminate", overwrite=True)
