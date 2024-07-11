from unitxt import add_to_catalog
from unitxt.blocks import (
    Task,
)

add_to_catalog(
    Task(
        inputs={
            "contexts": "List[str]",
            "contexts_ids": "List[int]",
            "dialog": "List[Dict[str,str]]",
        },
        outputs={"reference_answers": "List[str]"},
        metrics=[
            "metrics.f1_micro_multi_label",
        ],
        augmentable_inputs=["contexts"],
    ),
    "tasks.rag.response_generation_multi_turn_f1",
    overwrite=True,
)
