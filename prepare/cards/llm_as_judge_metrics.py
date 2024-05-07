from unitxt.blocks import TaskCard
from unitxt.catalog import add_to_catalog

card = TaskCard(
    loader=None,
    preprocess_steps=[],
    task="tasks.model_response_assessment.model_rating_single_turn",
    templates=["templates.model_response_assessment.mt_bench_model_rating_single_turn"],
)

add_to_catalog(
    card,
    "cards.rag.model_response_assessment.model_rating_single_turn",
    overwrite=True,
)

card = TaskCard(
    loader=None,
    preprocess_steps=[],
    task="tasks.model_response_assessment.model_rating_with_reference_single_turn",
    templates=[
        "templates.model_response_assessment.mt_bench_model_rating_with_reference_single_turn"
    ],
)

add_to_catalog(
    card,
    "cards.rag.model_response_assessment.model_rating_with_reference_single_turn",
    overwrite=True,
)
