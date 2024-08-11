from unitxt import add_to_catalog
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, Set
from unitxt.rag_metrics.binary_judge_metric import (
    GenerativeBinaryJudgeOpenAi,
    GenerativeBinaryJudgeWML,
)
from unitxt.rag_metrics.rag_metrics_utils import rag_fields, test_metric

metric_type_to_template = {
    "faithfulness": {
        "q_c_a": "judge_with_question_simplified_logprobs",
        "c_a": "judge_with_question_simplified_logprobs",
    },
    "context_relevance": {"q_c": "judge_context_relevance_ares_logprobs"},
    "correctness_holistic": {"q_c_a": "judge_correctness_simple_logprobs"},  # qca
    "correctness": {
        "q_c_a_gt": "judge_simplified_with_context_logprobs",
        "q_a_gt": "judge_simplified_no_context_logprobs",
    },
    "answer_relevance": {"q_a": "judge_answer_relevance_logprobs"},
}

model_names = {
    "meta-llama/llama-3-70b-instruct",
    "gpt-4o-mini",
    "mistralai/mixtral-8x7b-instruct-v01",
}


def add_judge_metrics():
    for judge_model_name in model_names:
        metric_class = (
            GenerativeBinaryJudgeOpenAi
            if judge_model_name.startswith("gpt-")
            else GenerativeBinaryJudgeWML
        )
        for (
            metric_type,
            input_fields_to_template_name,
        ) in metric_type_to_template.items():
            for (
                input_fields_str,
                template_name,
            ) in input_fields_to_template_name.items():
                metric_name = f"""{metric_type}_{input_fields_str}_judge_{judge_model_name.split("/")[-1].replace("-","_")}"""

                metric = metric_class(
                    main_score=metric_name,
                    model_name=judge_model_name,
                    task_name=f"tasks.rag_eval.{metric_type}.binary",
                    template_name=f"templates.rag_eval.{metric_type}.{template_name}",
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

                add_to_catalog(
                    metric_pipeline, name=f"metrics.rag.{metric_name}", overwrite=True
                )


add_judge_metrics()
