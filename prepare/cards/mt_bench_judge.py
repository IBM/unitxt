from unitxt.blocks import TaskCard
from unitxt.catalog import add_to_catalog

card = TaskCard(
    loader=None,
    preprocess_steps=[],
    task="tasks.rag.model_response_assessment",
    templates=[
        "templates.rag.model_response_assessment.llm_as_judge_using_mt_bench_template"
    ],
)

add_to_catalog(
    card,
    "cards.rag.model_response_assessment.llm_as_judge_using_mt_bench_template",
    overwrite=True,
)
