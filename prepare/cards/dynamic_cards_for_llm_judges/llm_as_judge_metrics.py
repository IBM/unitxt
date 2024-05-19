from unitxt.blocks import TaskCard
from unitxt.catalog import add_to_catalog

task = "tasks.dynamic_tasks_for_llm_judges.input_output_reference"
card = TaskCard(loader=None, preprocess_steps=[], task=task)
sub_task = ".".join(task.split(".")[-2:])
add_to_catalog(
    card,
    "cards.dynamic_cards_for_llm_judges.rating.single_turn",
    overwrite=True,
)
