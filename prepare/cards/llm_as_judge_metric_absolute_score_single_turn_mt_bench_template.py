from unitxt.blocks import TaskCard
from unitxt.catalog import add_to_catalog

card = TaskCard(
    loader=None,
    preprocess_steps=[],
    task="tasks.model_response_assessment.absolute_score_single_turn",
    templates=[
        "templates.model_response_assessment.mt_bench_absolute_score_single_turn"
    ],
)

add_to_catalog(
    card,
    "cards.rag.model_response_assessment.llm_as_judge_metric_absolute_score_single_turn",
    overwrite=True,
)
