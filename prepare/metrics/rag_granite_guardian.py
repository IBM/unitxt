from unitxt import add_to_catalog
from unitxt.metrics import GraniteGuardianRagRisk, MetricPipeline
from unitxt.operators import Copy, Set
from unitxt.string_operators import Join

rag_fields = ["ground_truths", "answer", "contexts", "question"]

test_examples = [
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": ["Supported foundation models available with watsonx.ai"],
    }
]

risk_names_to_pred_field = {
    "groundedness": "answer",
    "context_relevance": "question",
    "answer_relevance": "answer",
}


for granite_risk_name, pred_field in risk_names_to_pred_field.items():
    metric_name = f"""granite_guardian_{granite_risk_name}"""
    metric = GraniteGuardianRagRisk(
        main_score=metric_name,
        risk_name=granite_risk_name,
        user_message_field="question",
        assistant_message_field="answer",
    )

    metric_pipeline = MetricPipeline(
        main_score=metric_name,
        metric=metric,
        preprocess_steps=[
            Join(field="contexts", by="\n"),
            Copy(
                field_to_field={
                    field: f"task_data/{field if field != 'contexts' else 'context'}"
                    for field in rag_fields
                },
                not_exist_do_nothing=True,
            ),
            Copy(
                field_to_field={"prediction": f"task_data/{pred_field}"},
                not_exist_do_nothing=True,
            ),
            Set(
                fields={
                    "prediction": 0.0,  # these are not used by the metric
                    "references": [0.0],
                }
            ),
        ],
    )

    add_to_catalog(metric_pipeline, name=f"metrics.rag.{metric_name}", overwrite=True)
