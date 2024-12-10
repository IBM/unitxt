from unitxt import add_to_catalog
from unitxt.metrics import GraniteGuardianWMLMetric, MetricPipeline
from unitxt.operators import Copy, Set

rag_fields = ["ground_truths", "answer", "contexts", "question"]

test_examples = [
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": ["Supported foundation models available with watsonx.ai"],
    }
]

risk_names = ["groundedness", "context_relevance", "answer_relevance"]


for granite_risk_name in risk_names:
    metric_name = f"""granite_guardian_{granite_risk_name}"""
    metric = GraniteGuardianWMLMetric(
        main_score=metric_name,
        risk_name=granite_risk_name,
    )

    metric_pipeline = MetricPipeline(
        main_score=metric_name,
        metric=metric,
        preprocess_steps=[
            Copy(
                field_to_field={field: f"task_data/{field}" for field in rag_fields},
                not_exist_ok=True,
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
