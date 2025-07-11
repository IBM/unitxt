from unitxt.card import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Copy,
    ExecuteExpression,
    Move,
    Set,
)
from unitxt.splitters import RenameSplits
from unitxt.struct_data_operators import LoadJson
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="Salesforce/xlam-function-calling-60k",
        split="train",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        Set(fields={"dialog": [{"role": "user"}]}, use_deepcopy=True),
        Copy(field="query", to_field="dialog/0/content"),
        LoadJson(field="answers", to_field="reference_calls"),
        LoadJson(field="tools"),
        Move(field="tools/*/parameters", to_field="properties"),
        Set(fields={"tools/*/parameters": {"type": "object"}}, use_deepcopy=True),
        Copy(
            field="properties",
            to_field="tools/*/parameters/properties",
            set_every_value=True,
        ),
        ExecuteExpression(
            to_field="required",
            expression="[[p for p, c in tool['parameters']['properties'].items() if 'optional' not in c['type']] for tool in tools]",
        ),
        Copy(
            field="required",
            to_field="tools/*/parameters/required",
            set_every_value=True,
        ),
        "operators.fix_json_schema",
    ],
    task="tasks.tool_calling.multi_turn",
    templates=["templates.tool_calling.multi_turn"],
    __description__=(
        """This dataset contains 60,000 data points collected by APIGen, an automated data generation pipeline designed to produce verifiable high-quality datasets for function-calling applications. Each data point in the dataset is verified through three hierarchical stages: format checking, actual function executions, and semantic verification, ensuring its reliability and correctness."""
    ),
    __title__="""APIGen Function-Calling Datasets""",
    __tags__={
        "annotations_creators": "expert-generated",
        "language": ["en"],
        "license": "hf-gated",
        "size_categories": ["10K<n<100K"],
        "task_categories": [
            "question-answering",
            "reading-comprehension",
            "tool-calling",
            "multi-turn-tool-calling",
        ],
        "task_ids": [
            "tool-calling",
            "multi-turn-tool-calling",
            "reading-comprehension",
        ],
    },
)

# Test and add the card to the catalog
test_card(card, strict=False)
add_to_catalog(card, "cards.xlam_function_calling_60k", overwrite=True)
