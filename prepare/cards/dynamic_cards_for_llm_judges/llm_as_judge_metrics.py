from unitxt.blocks import TaskCard
from unitxt.catalog import add_to_catalog

tasks = [
    "tasks.response_assessment.rating.single_turn",
    "tasks.response_assessment.rating.single_turn_with_reference",
]
for task in tasks:
    card = TaskCard(loader=None, preprocess_steps=[], task=task)
    sub_task = ".".join(task.split(".")[-2:])
    add_to_catalog(
        card,
        f"cards.dynamic_cards_for_llm_judges.{sub_task}",
        overwrite=True,
    )
