template = "templates.llm_as_judge.model_response_assessment.mt_bench"
recipe = "cards.llm_as_judge.model_response_assessment"
model_type = "openai"
model_id = "gpt4"
"""
metric = LlmAsJudge(
    model_id=model_id, model_type=model_type, template=template, recipe=recipe
)

add_to_catalog(
    metric,
    "metrics.llm_as_judge.model_response_assessment.mt_bench_gpt4",
    overwrite=True,
)
"""
