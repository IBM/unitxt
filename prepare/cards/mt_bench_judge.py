from unitxt.blocks import TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=None,
    preprocess_steps=[
    ],
    task="tasks.llm_as_judge.model_response_assessment",
    templates=["templates.llm_as_judge.model_response_assessment.mt_bench"],
)

add_to_catalog(card, "cards.llm_as_judge.model_response_assessment.mt_bench", overwrite=True)



