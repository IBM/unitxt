from unitxt.blocks import TaskCard
from unitxt.catalog import add_to_catalog

# IMPORTANT: Unitxt currently does not support multi-turn judges.
task_list = [
    "tasks.response_assessment.rating.single_turn",
]

for task in task_list:
    card = TaskCard(loader=None, preprocess_steps=[], task=task)

    sub_task = task.split(".")[-1]
    add_to_catalog(
        card,
        f"cards.dynamic_cards_for_llm_judges.{sub_task}",
        overwrite=True,
    )
