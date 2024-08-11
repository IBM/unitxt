from unitxt import add_to_catalog
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, Set
from unitxt.rag_metrics.armorm_judge_metric import ArmoRMMetric
from unitxt.rag_metrics.rag_metrics_utils import rag_fields, test_metric

metric_type_to_template = {
    "correctness_holistic": {
        "q_c_a": {
            "template_name": "templates.rag.response_generation.answer_based_on_context",
            "task_name": "tasks.rag.response_generation",
        }
    },
    "answer_relevance": {
        "q_a": {
            "template_name": "templates.qa.open.simple2",
            "task_name": "tasks.qa.open",
        }
    },
}

for metric_type, input_fields_to_template_name in metric_type_to_template.items():
    for input_fields_str, template_task_dict in input_fields_to_template_name.items():
        metric_name = f"""{metric_type}_{input_fields_str}_ArmoRM"""

        metric = ArmoRMMetric(
            main_score=metric_name,
            task_name=template_task_dict["task_name"],
            template_name=template_task_dict["template_name"],
        )

        metric_pipeline = MetricPipeline(
            main_score=metric_name,
            metric=metric,
            preprocess_steps=[
                Copy(
                    field_to_field={
                        field: f"task_data/{field}" for field in rag_fields
                    },
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

        test_metric(metric_pipeline)

        add_to_catalog(metric_pipeline, artifact_name=f"metrics.rag.{metric_name}")
