from unitxt.blocks import LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import Copy
from unitxt.processors import ExtractWithRegex, PostProcess
from unitxt.string_operators import Replace
from unitxt.struct_data_operators import LoadJson
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="baharef/ToT", name="tot_semantic"),
    task=Task(
        input_fields={"prompt": str, "question": str},
        reference_fields={"label": str},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
    templates=[
        InputOutputTemplate(
            input_format="{prompt}",
            output_format='{{"answer": "{label}"}}',
            postprocessors=[
                PostProcess(
                    ExtractWithRegex(regex=r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')
                )
            ],
        )
    ],
    __tags__={
        "license": "cc-by-4.0",
        "language": ["en"],
        "task_categories": ["question-answering"],
    },
    __description__=(
        """Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning
ToT is a dataset designed to assess the temporal reasoning capabilities of AI models. It comprises two key sections:
ToT-semantic: Measuring the semantics and logic of time understanding.
ToT-arithmetic: Measuring the ability to carry out time arithmetic operations.
"""
    ),
)
test_card(card, strict=False)
add_to_catalog(card, "cards.tot.semantic", overwrite=True)

card = TaskCard(
    loader=LoadHF(path="baharef/ToT", name="tot_arithmetic"),
    preprocess_steps=[
        Replace(field="label", old="'", new='"'),
        LoadJson(field="label"),
        Copy(field="label/answer", to_field="label"),
    ],
    task=Task(
        input_fields={"question": str},
        reference_fields={"label": str},
        prediction_type=str,
        metrics=["metrics.accuracy"],
    ),
    templates=[
        InputOutputTemplate(
            input_format="{question}",
            output_format='{{"answer": "{label}"}}',
            postprocessors=[
                PostProcess(
                    ExtractWithRegex(regex=r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')
                )
            ],
        )
    ],
    __tags__={
        "license": "cc-by-4.0",
        "language": ["en"],
        "task_categories": ["question-answering"],
    },
    __description__=(
        """Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning
ToT is a dataset designed to assess the temporal reasoning capabilities of AI models. It comprises two key sections:
ToT-semantic: Measuring the semantics and logic of time understanding.
ToT-arithmetic: Measuring the ability to carry out time arithmetic operations.
"""
    ),
)
test_card(card, strict=False)
add_to_catalog(card, "cards.tot.arithmetic", overwrite=True)
